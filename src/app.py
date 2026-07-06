# Local Multimodal AI Chat - Multimodal chat application with local models
# Copyright (C) 2024 Leon Sander
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import streamlit as st
from chat_api_handler import ChatAPIHandler
from utils import get_timestamp, load_config, get_avatar
from audio_handler import transcribe_audio
from pdf_handler import add_documents_to_db
from html_templates import css
from database_operations import (
    get_db_manager,
    close_db_manager,
    DEFAULT_CHAT_MEMORY_LENGTH,
    DEFAULT_RETRIEVED_DOCUMENTS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP
)
from utils import list_openai_models, list_ollama_models, command
import sqlite3
config = load_config()

def toggle_pdf_chat():
    st.session_state.pdf_chat = True
    clear_cache()

def detoggle_pdf_chat():
    st.session_state.pdf_chat = False

def get_session_key():
    if st.session_state.session_key == "new_session":
        st.session_state.new_session_key = get_timestamp()
        return st.session_state.new_session_key
    return st.session_state.session_key

def delete_chat_session_history():
    db_manager = get_db_manager()
    db_manager.message_repo.delete_chat_history(st.session_state.session_key)
    st.session_state.session_index_tracker = "new_session"

def clear_cache():
    st.cache_resource.clear()

def list_model_options():
    if st.session_state.endpoint_to_use == "ollama":
        ollama_options = list_ollama_models()
        if ollama_options == []:
            st.warning("No ollama models available, please choose one from https://ollama.com/library and pull with /pull <model_name>")
        return ollama_options
    elif st.session_state.endpoint_to_use == "openai":
        return list_openai_models()

def update_model_options():
    st.session_state.model_options = list_model_options()

def main():
    st.title("Multimodal Local Chat App")
    st.write(css, unsafe_allow_html=True)
    
    if "db_manager" not in st.session_state:
        st.session_state.session_key = "new_session"
        st.session_state.new_session_key = None
        st.session_state.session_index_tracker = "new_session"
        st.session_state.audio_uploader_key = 0
        st.session_state.pdf_uploader_key = 1
        st.session_state.endpoint_to_use = "ollama"
        st.session_state.model_options = list_model_options()
        st.session_state.model_tracker = None

    if st.session_state.session_key == "new_session" and st.session_state.new_session_key != None:
        st.session_state.session_index_tracker = st.session_state.new_session_key
        st.session_state.new_session_key = None

    db_manager = get_db_manager()
    
    st.sidebar.title("Chat Sessions")
    chat_sessions = ["new_session"] + db_manager.message_repo.get_all_chat_history_ids()
    try:
        index = chat_sessions.index(st.session_state.session_index_tracker)
    except ValueError:
        st.session_state.session_index_tracker = "new_session"
        index = chat_sessions.index(st.session_state.session_index_tracker)
        clear_cache()

    st.sidebar.selectbox("Select a chat session", chat_sessions, key="session_key", index=index)
    
    # Add configuration section
    st.sidebar.title("Configuration")
    
    # Model Settings
    st.sidebar.subheader("Model Configuration")
    api_col, model_col = st.sidebar.columns(2)
    api_col.selectbox(label="Select an API", options = ["ollama","openai"], key="endpoint_to_use", on_change=update_model_options)
    model_col.selectbox(label="Select a Model", options = st.session_state.model_options, key="model_to_use")
    pdf_toggle_col, voice_rec_col = st.sidebar.columns(2)
    pdf_toggle_col.toggle("PDF Chat", key="pdf_chat", value=False, on_change=clear_cache)
    
    # Add audio input in the sidebar
    audio_input = st.sidebar.audio_input("Record Audio", key="audio_input")
    
    delete_chat_col, clear_cache_col = st.sidebar.columns(2)
    delete_chat_col.button("Delete Chat Session", on_click=delete_chat_session_history)

    # Chat History Settings
    st.sidebar.subheader("Chat History")
    chat_memory_length = st.sidebar.number_input(
        "Number of Previous Messages",
        value=int(db_manager.settings_repo.get_setting("chat_memory_length", DEFAULT_CHAT_MEMORY_LENGTH)),
        key="chat_memory_length",
        on_change=lambda: db_manager.settings_repo.update_setting("chat_memory_length", st.session_state.chat_memory_length)
    )
    
    # PDF Processing Settings
    st.sidebar.subheader("PDF Processing")
    retrieved_docs = st.sidebar.number_input(
        "Number of Retrieved PDF Chunks",
        value=int(db_manager.settings_repo.get_setting("retrieved_documents", DEFAULT_RETRIEVED_DOCUMENTS)),
        key="retrieved_documents",
        on_change=lambda: db_manager.settings_repo.update_setting("retrieved_documents", st.session_state.retrieved_documents)
    )
    chunk_size = st.sidebar.number_input(
        "PDF Chunk Size (characters)",
        value=int(db_manager.settings_repo.get_setting("chunk_size", DEFAULT_CHUNK_SIZE)),
        key="chunk_size",
        on_change=lambda: db_manager.settings_repo.update_setting("chunk_size", st.session_state.chunk_size)
    )
    chunk_overlap = st.sidebar.number_input(
        "PDF Chunk Overlap (characters)",
        value=int(db_manager.settings_repo.get_setting("chunk_overlap", DEFAULT_CHUNK_OVERLAP)),
        key="chunk_overlap",
        on_change=lambda: db_manager.settings_repo.update_setting("chunk_overlap", st.session_state.chunk_overlap)
    )
    
    chat_container = st.container()
    
    # Replace file uploaders with chat_input
    user_input = st.chat_input(
        "Type your message here",
        key="user_input",
        accept_file=True,
        file_type=["pdf", "jpg", "jpeg", "png", "wav", "mp3", "ogg"]
    )

    if user_input:
        # Handle file uploads if present
        if user_input.files:
            # Group files by type
            pdf_files = []
            image_files = []
            audio_files = []
            
            for file in user_input.files:
                file_type = file.name.split('.')[-1].lower()
                if file_type in ["pdf"]:
                    pdf_files.append(file)
                elif file_type in ["jpg", "jpeg", "png"]:
                    image_files.append(file)
                elif file_type in ["wav", "mp3", "ogg"]:
                    audio_files.append(file)
            
            # Process PDFs in batch
            if pdf_files:
                with st.spinner("Processing pdfs..."):
                    add_documents_to_db(pdf_files)
                    # If there's a text message, process it after PDFs are added
                    if user_input.text:
                        chat_history = db_manager.message_repo.load_last_k_text_messages(get_session_key(), st.session_state.chat_memory_length)
                        llm_answer = ChatAPIHandler.chat(user_input=user_input.text, chat_history=chat_history)
                        db_manager.message_repo.save_message(get_session_key(), "user", "text", user_input.text)
                        db_manager.message_repo.save_message(get_session_key(), "assistant", "text", llm_answer)
            
            # Process images
            if image_files:
                with st.spinner("Processing images..."):
                    for image_file in image_files:
                        llm_answer = ChatAPIHandler.chat(user_input=user_input.text or "", chat_history=[], image=image_file.getvalue())
                        db_manager.message_repo.save_message(get_session_key(), "user", "text", user_input.text or "")
                        db_manager.message_repo.save_message(get_session_key(), "user", "image", image_file.getvalue())
                        db_manager.message_repo.save_message(get_session_key(), "assistant", "text", llm_answer)
            
            # Process audio files
            if audio_files:
                for audio_file in audio_files:
                    transcribed_audio = transcribe_audio(audio_file.getvalue())
                    llm_answer = ChatAPIHandler.chat(user_input=(user_input.text or "") + "\n" + transcribed_audio, chat_history=[])
                    db_manager.message_repo.save_message(get_session_key(), "user", "text", user_input.text or "")
                    db_manager.message_repo.save_message(get_session_key(), "user", "audio", audio_file.getvalue())
                    db_manager.message_repo.save_message(get_session_key(), "assistant", "text", llm_answer)
        
        # Handle text input only if no files were processed
        elif user_input.text:
            if user_input.text.startswith("/"):
                response = command(user_input.text)
                db_manager.message_repo.save_message(get_session_key(), "user", "text", user_input.text)
                db_manager.message_repo.save_message(get_session_key(), "assistant", "text", response)
            else:
                chat_history = db_manager.message_repo.load_last_k_text_messages(get_session_key(), st.session_state.chat_memory_length)
                llm_answer = ChatAPIHandler.chat(user_input=user_input.text, chat_history=chat_history)
                db_manager.message_repo.save_message(get_session_key(), "user", "text", user_input.text)
                db_manager.message_repo.save_message(get_session_key(), "assistant", "text", llm_answer)

    # Handle audio input
    if audio_input:
        transcribed_audio = transcribe_audio(audio_input.getvalue())
        chat_history = db_manager.message_repo.load_last_k_text_messages(get_session_key(), st.session_state.chat_memory_length)
        llm_answer = ChatAPIHandler.chat(user_input=transcribed_audio, chat_history=chat_history)
        db_manager.message_repo.save_message(get_session_key(), "user", "audio", audio_input.getvalue())
        db_manager.message_repo.save_message(get_session_key(), "assistant", "text", llm_answer)

    if (st.session_state.session_key != "new_session") != (st.session_state.new_session_key != None):
        with chat_container:
            chat_history_messages = db_manager.message_repo.load_messages(get_session_key())

            for message in chat_history_messages:
                with st.chat_message(name=message["sender_type"], avatar=get_avatar(message["sender_type"])):
                    if message["message_type"] == "text":
                        st.write(message["content"])
                    if message["message_type"] == "image":
                        st.image(message["content"])
                    if message["message_type"] == "audio":
                        st.audio(message["content"], format="audio/wav")

        if (st.session_state.session_key == "new_session") and (st.session_state.new_session_key != None):
            st.rerun()

if __name__ == "__main__":
    main()
