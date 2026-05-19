# BSD 3-Clause License
#
# Copyright (c) 2026, Jesús Daniel Colmenares Oviedo <DtxdF@disroot.org>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import asyncio
import base64
import contextlib
import datetime
import tempfile
import glob
import random
import json
import os
import logging
from typing import Dict, List

import aiofiles
import hjson
import mcp
import nh3
import openai
import agents as openai_agents
import aiogram.utils.chat_action
import google.genai
import lmdb
import magic
import pydantic

MCP_CONFIG = None
MCP_CONFIG_MTIME = 0
DB = None
TOOLS = {}
HOOKS = {}
DELAYS = {}

logger = logging.getLogger(__name__)

class MCPTimeoutsSchema(pydantic.BaseModel):
    timeout: int = 10
    connect_timeout_seconds: int = 10
    cleanup_timeout_seconds: int = 10

class MCPServersSchema(pydantic.BaseModel):
    enabled: bool = True
    command: str
    args: List[str]  = []
    timeout: int = 10
    env: Dict[str, str] = {}
    cwd: str = os.getcwd()

class MCPConfigSchema(pydantic.BaseModel):
    mcpTimeouts: MCPTimeoutsSchema = MCPTimeoutsSchema()
    mcpServers: Dict[str, MCPServersSchema]

async def make_response(message, bot=None, text=None):
    async with aiogram.utils.chat_action.ChatActionSender.typing(bot=message.bot, chat_id=message.chat.id):
        if not _is_manager():
            user_id = message.from_user.id

            if not DELAYS.get(user_id, False):
                DELAYS[user_id] = True

            else:
                await _delay()

                DELAYS[user_id] = False

        if os.getenv("POWERBOT_GEMINI_API_KEY") is not None:
            await make_response_gemini(message, bot, text)

        else:
            await make_response_openai(message, bot, text)

async def _delay():
    delay = os.getenv("POWERBOT_DELAY")

    if delay is None:
        return

    delay = float(delay)

    logger.debug("(seconds:%f) Delaying ...", delay)

    await asyncio.sleep(delay)

async def make_response_openai(message, bot=None, text=None):
    base_url = os.getenv("POWERBOT_OPENAI_BASE_URL")
    api_key = os.getenv("POWERBOT_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    model = os.getenv("POWERBOT_OPENAI_MODEL")
    vision_model = os.getenv("POWERBOT_OPENAI_VISION_MODEL")

    client = openai.AsyncOpenAI(
        base_url=base_url,
        api_key=api_key
    )

    user_id = message.from_user.id

    system_instruction = os.getenv("POWERBOT_SYSTEM_INSTRUCTION")

    data = {
        "AIOGRAM_MESSAGE" : message,
        "AIOGRAM_BOT" : bot,
        "OPENAI_CLIENT" : client
    }

    allow_continue = await call_hooks(data)

    if not allow_continue:
        return

    tools = get_tools(data)
    tools = [openai_agents.function_tool(f) for f in tools]

    mcp_servers = await get_mcp_servers_openai()

    mcp_timeouts = await get_mcp_timeouts()
    mcp_params = {
        "connect_timeout_seconds" : mcp_timeouts["connect_timeout_seconds"],
        "cleanup_timeout_seconds" : mcp_timeouts["cleanup_timeout_seconds"],
    }

    async with openai_agents.mcp.MCPServerManager(mcp_servers, **mcp_params) as mcp_manager:
        system_instruction = os.getenv("POWERBOT_SYSTEM_INSTRUCTION")

        model_settings = openai_agents.ModelSettings(
            temperature=float(os.getenv("POWERBOT_TEMPERATURE")),
            top_p=float(os.getenv("POWERBOT_TOP_P")),
            max_tokens=int(os.getenv("POWERBOT_MAX_TOKENS")),
            frequency_penalty=float(os.getenv("POWERBOT_FREQUENCY_PENALTY")),
            presence_penalty=float(os.getenv("POWERBOT_PRESENCE_PENALTY"))
        )

        files, has_vision = await process_file_openai(client, message, bot)

        if has_vision:
            vision_model = openai_agents.OpenAIChatCompletionsModel(
                model=vision_model,
                openai_client=client
            )

            agent = openai_agents.Agent(
                name="powerbot",
                instructions=system_instruction,
                tools=tools,
                model=vision_model,
                model_settings=model_settings,
                mcp_servers=mcp_manager.active_servers or []
            )

        else:
            model = openai_agents.OpenAIChatCompletionsModel(
                model=model,
                openai_client=client
            )

            agent = openai_agents.Agent(
                name="powerbot",
                instructions=system_instruction,
                tools=tools,
                model=model,
                model_settings=model_settings,
                mcp_servers=mcp_manager.active_servers
            )

        full_context = load_history_openai(user_id)
        full_context = purge_history_openai(full_context)

        text = text or message.caption or message.text or ""
        text = _limit_length(text)

        if len(files) == 0 and len(text.strip()) == 0:
            response = await openai_agents.Runner.run(
                starting_agent=agent,
                input="[POWERBOT: empty]"
            )

        else:
            current_content = [{
                "type" : "text",
                "text" : text
            }]

            if message.reply_to_message is not None and \
                    message.reply_to_message.text is not None:
                current_content.append({
                    "type" : "text",
                    "text" : _limit_length(f"[POWERBOT: reply]: {message.reply_to_message.text}")
                })

            if len(files) > 0:
                current_content.extend(files)

            full_context.append({
                "role" : "user",
                "content" : current_content
            })

            response = await openai_agents.Runner.run(
                starting_agent=agent,
                input=full_context
            )

        output = _escape_bad_output(response.final_output)

        if len(output.strip()) == 0:
            response = await openai_agents.Runner.run(
                starting_agent=agent,
                input="[POWERBOT: empty-response]"
            )
            output = _escape_bad_output(response.final_output)

        else:
            full_context.append({
                "role" : "assistant",
                "content" : output
            })

            full_context = purge_history_openai(full_context)

            save_history_openai(user_id, full_context)

        if len(output.strip()) == 0:
            return

        await message.answer(output)

async def get_mcp_servers_openai():
    servers = []

    mcp_servers = await get_mcp_servers()

    if mcp_servers is None:
        return servers

    for name, params in mcp_servers.items():
        enabled = params["enabled"]

        if not enabled:
            continue

        try:
            mcp_server = openai_agents.mcp.MCPServerStdio(
                name=name,
                params={
                    "command" : params["command"],
                    "args" : params.get("args"),
                    "env" : params.get("env"),
                    "cwd" : params.get("cwd")
                },
                client_session_timeout_seconds=params.get("timeout")
            )

        except Exception as err:
            error_type = err.__class__.__name__
            error_message = str(err)

            logger.exception("(exception:%s, server:%s) exception while processing an MCP server: %s", error_type, name, error_message)

            continue

        servers.append(mcp_server)

    return servers

async def get_mcp_timeouts():
    mcp_config = await get_mcp_config()

    if mcp_config is None:
        mcp_timeouts = MCPTimeoutsSchema()
        mcp_timeouts = mcp_timeouts.model_dump()

    else:
        mcp_timeouts = mcp_config["mcpTimeouts"]

    return mcp_timeouts

async def get_mcp_servers():
    mcp_config = await get_mcp_config()

    if mcp_config is None:
        return

    return mcp_config["mcpServers"]

async def get_mcp_config():
    global MCP_CONFIG
    global MCP_CONFIG_MTIME

    mcp_config = os.getenv("POWERBOT_MCP")

    if mcp_config is None or not os.path.isfile(mcp_config):
        return

    mtime = os.path.getmtime(mcp_config)

    if mtime == MCP_CONFIG_MTIME:
        return MCP_CONFIG

    MCP_CONFIG_MTIME = mtime

    async with aiofiles.open(mcp_config) as fd:
        try:
            content = await fd.read()

            config = hjson.loads(content)
            config = MCPConfigSchema(**config)
            MCP_CONFIG = config.model_dump()

        except Exception as err:
            error_type = err.__class__.__name__
            error_message = str(err)

            logger.exception("(exception:%s) exception while processing MCP configuration file: %s", error_type,  error_message)

        return MCP_CONFIG

async def process_file_openai(client, message, bot):
    document = message.document

    if document is None and message.reply_to_message is not None:
        document = message.reply_to_message.document

    photo = message.photo

    if photo is None and message.reply_to_message is not None:
        photo = message.reply_to_message.photo

    audio = message.audio

    if audio is None and message.reply_to_message is not None:
        audio = message.reply_to_message.audio

    video = message.video

    if video is None and message.reply_to_message is not None:
        video = message.reply_to_message.video

    voice = message.voice

    if voice is None and message.reply_to_message is not None:
        voice = message.reply_to_message.voice

    sticker = message.sticker

    if sticker is None and message.reply_to_message is not None:
        sticker = message.reply_to_message.sticker

    video_note = message.video_note

    if video_note is None and message.reply_to_message is not None:
        video_note = message.reply_to_message.video_note

    animation = message.animation

    if animation is None and message.reply_to_message is not None:
        animation = message.reply_to_message.animation

    contents = []
    has_vision = False

    if document is None and \
            photo is None and \
            audio is None and \
            video is None and \
            voice is None and \
            sticker is None and \
            video_note is None and \
            animation is None:
        return contents, has_vision

    if video is not None or \
            video_note is not None or \
            sticker is not None or \
            animation is not None:
        if video is not None:
            file_obj = video

        elif video_note is not None:
            file_obj = video_note

        elif sticker is not None:
            file_obj = sticker

        elif animation is not None:
            file_obj = animation

        file_id = file_obj.file_id

        with tempfile.NamedTemporaryFile(prefix="powerbot", mode="wb") as fd:
            file = await bot.get_file(file_id)

            await bot.download_file(file.file_path, destination=fd.name)

            try:
                frames = await extract_video_frames(fd.name)

                if len(frames) == 0:
                    frames = await extract_video_frames(fd.name, static=True)

            except Exception as err:
                error_type = err.__class__.__name__
                error_message = str(err)

                logger.exception("(exception:%s, file:%s) exception while processing a video: %s", error_type, fd.name, error_message)

                contents.append({
                    "type" : "text",
                    "text" : "[POWERBOT: video-error]"
                })

                return contents, has_vision

            logger.debug("(frames:%d) encoding frames ...", len(frames))

            for frame in frames:
                img = base64.b64encode(frame)
                img = img.decode()

                contents.append({
                    "type" : "image_url",
                    "image_url" : {
                        "url" : f"data:image/jpeg;base64,{img}"
                    }
                })

            has_vision = True

    elif photo is not None:
        file_obj = photo[-1]
        file_id = file_obj.file_id

        file = await bot.get_file(file_id)
        buff = await bot.download(file)

        img = base64.b64encode(buff.getvalue())
        img = img.decode()

        contents.append({
            "type" : "image_url",
            "image_url" : {
                "url" : f"data:image/jpeg;base64,{img}"
            }
        })

        has_vision = True

    elif audio is not None or voice is not None:
        if audio is not None:
            file_obj = audio

        else:
            file_obj = voice

        file_id = file_obj.file_id

        file = await bot.get_file(file_id)
        buff = await bot.download(file)
        buff.name = "audio.ogg"

        transcript = await client.audio.transcriptions.create(
            model=os.getenv("POWERBOT_OPENAI_TRANSCRIPTIONS_MODEL"),
            file=buff,
            response_format="text"
        )

        contents.append({
            "type" : "text",
            "text" : f"[POWERBOT: transcription]"
        })

        contents.append({
            "type" : "text",
            "text" : _limit_length(transcript)
        })

    elif document is not None:
        file_obj = document
        file_id = file_obj.file_id

        with tempfile.NamedTemporaryFile(prefix="powerbot", mode="wb") as fd:
            file = await bot.get_file(file_id)

            await bot.download_file(file.file_path, destination=fd.name)

            try:
                chunks = await extract_document_text(fd.name)

            except Exception as err:
                error_type = err.__class__.__name__
                error_message = str(err)

                logger.exception("(exception:%s, file:%s) exception while processing a document: %s", error_type, fd.name, error_message)

                contents.append({
                    "type" : "text",
                    "text" : "[POWERBOT: document-error]"
                })

                return contents, has_vision

            contents.append({
                "type" : "text",
                "text" : f"[POWERBOT: document]"
            })

            for chunk in chunks:
                contents.append({
                    "type" : "text",
                    "text" : chunk
                })

    return contents, has_vision

async def extract_document_text(pathname):
    extension = _mime2ext(pathname)

    cmd = [
        "textract",
        "--extension", extension,
    ]

    if extension == "pdf":
        cmd.extend(["--method", "pdfminer"])

    cmd.extend(["--", pathname])

    chunk_size = int(os.getenv("POWERBOT_TEXT_CHUNK_SIZE"))
    max_chunks = int(os.getenv("POWERBOT_TEXT_MAX_CHUNKS"))

    logger.debug("(document:%s, chunk_size:%d) processing document ...", pathname, chunk_size)

    process = await asyncio.create_subprocess_exec(*cmd,
        stdout=asyncio.subprocess.PIPE
    )

    (text, _) = await process.communicate()

    logger.debug("(document:%s, chunk_size:%d, rc:%d) processed", pathname, chunk_size, process.returncode)

    assert process.returncode == 0

    text = text.decode()

    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    if len(chunks) > max_chunks:
        step = len(chunks) / max_chunks
        selected_chunks = [chunks[int(i * step)] for i in range(max_chunks)]

        chunks = selected_chunks

    return chunks

def _mime2ext(pathname):
    table = {
        'text/csv': 'csv',
        'application/msword': 'doc',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
        'message/rfc822': 'eml',
        'application/epub+zip': 'epub',
        'text/html': 'html',
        'application/xhtml+xml': 'html',
        'application/json': 'json',
        'application/vnd.ms-outlook': 'msg',
        'application/vnd.oasis.opendocument.text': 'odt',
        'application/pdf': 'pdf',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'pptx',
        'application/postscript': 'ps',
        'text/rtf': 'rtf',
        'application/x-font-ttf': 'tff',
        'application/vnd.ms-excel': 'xls',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx'
    }

    mime_type = magic.from_file(pathname, mime=True)

    if mime_type not in table:
        return "txt"

    return table[mime_type]

async def extract_video_frames(pathname, static=False):
    max_frames = int(os.getenv("POWERBOT_VIDEO_MAX_FRAMES"))

    logger.debug("(video:%s, max_frames:%d) processing video ...", pathname, max_frames)

    with tempfile.TemporaryDirectory(prefix="powerbot") as tmpdir_fd:
        output = os.path.join(tmpdir_fd, "frame_%04d.jpeg")
        vf_args = "fps=1,scale=512:512:force_original_aspect_ratio=decrease,format=yuvj420p"

        if static:
            vf_args = "scale=512:512:force_original_aspect_ratio=decrease,format=yuvj420p"

        cmd = [
            "ffmpeg", "-y", "-i", pathname,
            "-vf", vf_args,
            "-q:v", "5",
            "-threads", "0",
            "-loglevel", "error",
            output
        ]

        process = await asyncio.create_subprocess_exec(*cmd,
            stdout=asyncio.subprocess.DEVNULL
        )

        await process.communicate()

        logger.debug("(video:%s, max_frames:%d, rc:%d) processed", pathname, max_frames, process.returncode)

        assert process.returncode == 0

        frames = sorted(glob.glob(f"{tmpdir_fd}/frame_*.jpeg"))

        if len(frames) > max_frames:
            step = len(frames) / max_frames
            selected_frames = [frames[int(i * step)] for i in range(max_frames)]

            frames = selected_frames

        content = []

        for frame in frames:
            async with aiofiles.open(frame, "rb") as fd:
                content.append(await fd.read())

        return content

def save_history_openai(user_id, history):
    logger.debug("(user:%d) saving history ...", user_id)

    db = make_db()

    with DB.begin(write=True) as txn:
        key = _make_history_key("openai", user_id)
        key = key.encode()

        value = json.dumps(history)
        value = value.encode()

        txn.put(key, value)

def load_history_openai(user_id):
    db = make_db()

    history = []

    with DB.begin() as txn:
        key = _make_history_key("openai", user_id)
        raw = txn.get(key.encode())

        if raw is not None:
            history = json.loads(raw.decode())

    return history

def purge_history_openai(history):
    _history = []

    history_limit = int(os.getenv("POWERBOT_CHAT_HISTORY"))
    history_limit = abs(history_limit)

    history = history[-history_limit:]

    if len(history) > 0 and history[0]["role"] == "assistant":
        history.pop(0)

    for item in history:
        if item["role"] == "user":
            item["content"] = [c for c in item["content"] if c["type"] == "text"]

        _history.append(item)

    return _history

async def make_response_gemini(message, bot=None, text=None):
    api_key = os.getenv("POWERBOT_GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
    model = os.getenv("POWERBOT_GEMINI_MODEL")

    client = google.genai.Client(
        api_key=api_key
    )
    
    user_id = message.from_user.id

    chat = client.aio.chats.create(
        model=model,
        history=load_history_gemini(user_id)
    )

    data = {
        "AIOGRAM_MESSAGE" : message,
        "AIOGRAM_BOT" : bot,
        "GEMINI_CLIENT" : client,
        "GEMINI_CHAT" : chat
    }

    allow_continue = await call_hooks(data)

    if not allow_continue:
        return

    system_instruction = os.getenv("POWERBOT_SYSTEM_INSTRUCTION")

    contents = []

    tools = get_tools(data)

    text = text or message.caption or message.text or ""
    text = _limit_length(text)

    contents.append(text)

    if message.reply_to_message is not None and \
            message.reply_to_message.text is not None:
        contents.append(
            _limit_length(f"[POWERBOT: reply]: {message.reply_to_message.text}")
        )

    mcp_timeouts = await get_mcp_timeouts()
    mcp_timeout = mcp_timeouts["timeout"]

    async with contextlib.AsyncExitStack() as stack:
        mcp_sessions = []
        mcp_config = await get_mcp_servers_gemini()

        for mcp_name, mcp_params in mcp_config.items():
            try:
                read, write = await stack.enter_async_context(mcp.client.stdio.stdio_client(mcp_params))
                session = await stack.enter_async_context(
                    mcp.ClientSession(
                        read,
                        write,
                        read_timeout_seconds=datetime.timedelta(seconds=mcp_timeout)
                    )
                )

                await session.initialize()

                mcp_sessions.append(session)

            except Exception as err:
                error_type = err.__class__.__name__
                error_message = str(err)

                logger.exception("(exception:%s, server:%s) exception while connecting to an MCP server: %s", error_type, mcp_name, error_message)

        tools += mcp_sessions

        file = None

        if bot is not None:
            file = await process_file_gemini(client, message, bot)

            if file is not None:
                contents.append(file)

        config = google.genai.types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=tools,
            automatic_function_calling=google.genai.types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=int(os.getenv("POWERBOT_GEMINI_AFC_LIMIT"))
            ),
            temperature=float(os.getenv("POWERBOT_TEMPERATURE")),
            top_p=float(os.getenv("POWERBOT_TOP_P")),
            top_k=float(os.getenv("POWERBOT_TOP_K")),
            max_output_tokens=int(os.getenv("POWERBOT_MAX_TOKENS")),
            frequency_penalty=float(os.getenv("POWERBOT_FREQUENCY_PENALTY")),
            presence_penalty=float(os.getenv("POWERBOT_PRESENCE_PENALTY"))
        )

        if file is None and len(contents[0].strip()) == 0:
            response = await chat.send_message(
                message="[POWERBOT: empty]",
                config=config
            )

        else:
            response = await chat.send_message(
                message=contents, config=config
            )

            if file is not None:
                await client.aio.files.delete(name=file.name)

        history = chat.get_history()

        save_history_gemini(user_id, history)

        output = _escape_bad_output(response.text)

        if len(output.strip()) == 0:
            response = await chat.send_message(
                message="[POWERBOT: empty-response]",
                config=config
            )

            output = _escape_bad_output(response.text)

        if len(output.strip()) == 0:
            return

        await message.answer(output)

async def get_mcp_servers_gemini():
    servers = {}

    mcp_servers = await get_mcp_servers()

    if mcp_servers is None:
        return servers

    for name, params in mcp_servers.items():
        enabled = params.get("enabled", True)

        if not enabled:
            continue

        try:
            mcp_server = mcp.StdioServerParameters(
                command=params["command"],
                args=params.get("args", []),
                env=params.get("env", {}),
                cwd=params.get("cwd", os.getcwd())
            )

        except Exception as err:
            error_type = err.__class__.__name__
            error_message = str(err)

            logger.exception("(exception:%s, server:%s) exception while processing an MCP server: %s", error_type, name, error_message)

            continue

        servers[name] = mcp_server

    return servers

async def process_file_gemini(client, message, bot):
    document = message.document

    if document is None and message.reply_to_message is not None:
        document = message.reply_to_message.document

    photo = message.photo

    if photo is None and message.reply_to_message is not None:
        photo = message.reply_to_message.photo

    audio = message.audio

    if audio is None and message.reply_to_message is not None:
        audio = message.reply_to_message.audio

    video = message.video

    if video is None and message.reply_to_message is not None:
        video = message.reply_to_message.video

    voice = message.voice

    if voice is None and message.reply_to_message is not None:
        voice = message.reply_to_message.voice

    sticker = message.sticker

    if sticker is None and message.reply_to_message is not None:
        sticker = message.reply_to_message.sticker

    video_note = message.video_note

    if video_note is None and message.reply_to_message is not None:
        video_note = message.reply_to_message.video_note

    animation = message.animation

    if animation is None and message.reply_to_message is not None:
        animation = message.reply_to_message.animation

    if document is None and \
            photo is None and \
            audio is None and \
            video is None and \
            voice is None and \
            sticker is None and \
            video_note is None and \
            animation is None:
        return

    mime_type = None

    if photo is not None:
        file_obj = photo[-1]
        file_id = file_obj.file_id
        file_name = f"photo_{file_id}"

    elif audio is not None:
        file_obj = audio
        file_id = file_obj.file_id
        file_name = file_obj.file_name or f"audio_{file_id}"
        mime_type = file_obj.mime_type

    elif video is not None:
        file_obj = video
        file_id = file_obj.file_id
        file_name = file_obj.file_name or f"video_{file_id}"
        mime_type = file_obj.mime_type

    elif voice is not None:
        file_obj = voice
        file_id = file_obj.file_id
        file_name = f"voice_{file_id}.ogg"

    elif sticker is not None:
        file_obj = sticker
        file_id = file_obj.file_id
        file_name = f"sticker_{file_id}"

    elif video_note is not None:
        file_obj = video_note
        file_id = file_obj.file_id
        file_name = f"video_note_{file_id}"

    elif animation is not None:
        file_obj = animation
        file_id = file_obj.file_id
        file_name = f"animation_{file_id}"
        mime_type = file_obj.mime_type

    elif document is not None:
        file_obj = document
        file_id = file_obj.file_id
        file_name = file_obj.file_name
        mime_type = file_obj.mime_type

    with tempfile.NamedTemporaryFile(prefix="powerbot", mode="wb") as fd:
        file = await bot.get_file(file_id)

        await bot.download_file(file.file_path, destination=fd.name)

        if mime_type is None:
            mime_type = magic.from_file(fd.name, mime=True)

        uploaded_file = await client.aio.files.upload(
            file=fd.name,
            config=google.genai.types.UploadFileConfig(
                display_name=file_name,
                mime_type=mime_type
            )
        )

        n = 0

        while uploaded_file.state == "PROCESSING":
            n += random.randint(1, 3)

            await asyncio.sleep(n)

            if n >= 30:
                n = 0

            uploaded_file = await client.aio.files.get(name=uploaded_file.name)

        return uploaded_file

def save_history_gemini(user_id, history):
    history_limit = int(os.getenv("POWERBOT_CHAT_HISTORY"))
    history_limit = abs(history_limit)

    history = [h.to_json_dict() for h in history]
    history = history[-history_limit:]

    if len(history) > 0 and history[0]["role"] == "model":
        history.pop(0)

    # Purge non-text data to avoid expiration errors.
    for index, item in enumerate(history):
        parts = [p for p in item["parts"] if "text" in p]

        history[index]["parts"] = parts

    logger.debug("(user:%d) saving history ...", user_id)

    db = make_db()

    with DB.begin(write=True) as txn:
        key = _make_history_key("gemini", user_id)
        key = key.encode()

        value = json.dumps(history)
        value = value.encode()

        txn.put(key, value)

def load_history_gemini(user_id):
    db = make_db()

    history = []

    with DB.begin() as txn:
        key = _make_history_key("gemini", user_id)
        raw = txn.get(key.encode())

        if raw is not None:
            history = json.loads(raw.decode())

    return history

def _make_history_key(provider, user_id):
    if _is_manager():
        user_type = "manager"

    else:
        user_type = "user"

    key = f"history_{provider}_{user_type}_{user_id}"

    return key

def _is_manager():
    is_manager = os.getenv("POWERBOT_IS_MANAGER", "0")

    return is_manager != "0"

def make_db():
    global DB

    if DB is None:
        db_path = os.getenv("POWERBOT_LMDB_PATH")
        map_size = int(os.getenv("POWERBOT_LMDB_SIZE"))
        DB = lmdb.open(db_path, map_size=map_size)

    return DB

async def call_hooks(data):
    hooks = get_hooks(data)

    hook_output = None

    for hook in hooks:
        try:
            hook_output = await hook(hook_output)

        except Exception as err:
            error_type = err.__class__.__name__
            error_message = str(err)

            logger.exception("(exception:%s) exception while processing a hook: %s", error_type, error_message)

    allow_continue = hook_output is None

    return allow_continue

def get_hooks(data):
    hooksdir = os.getenv("POWERBOT_HOOKS")

    hooks = get_tools(data,
        toolsdirs=hooksdir,
        tools_mapping=HOOKS
    )

    hooks = sorted(hooks, key=lambda f: f.__name__)

    return hooks

def get_tools(data, toolsdirs=None, tools_mapping=None):
    if toolsdirs is None:
        toolsdirs = os.getenv("POWERBOT_TOOLS")

    if toolsdirs is None:
        return []

    if tools_mapping is None:
        tools_mapping = TOOLS

    if _is_manager():
        toolsdir = os.path.join(toolsdirs, "manager")

    else:
        toolsdir = os.path.join(toolsdirs, "user")

    if not os.path.isdir(toolsdir):
        return []

    for tool in list(tools_mapping):
        toolpath = os.path.join(toolsdir, tool)

        if not os.path.isfile(toolpath):
            logger.warning("(tool:%s, directory:%s) removing ...", tool, toolsdir)

            del tools_mapping[tool]

    tools = []

    for tool in os.listdir(toolsdir):
        toolpath = os.path.join(toolsdir, tool)

        try:
            toolinf = dynload(tool, toolpath, data, tools_mapping)

        except Exception as err:
            error_type = err.__class__.__name__
            error_message = str(err)

            logger.exception("(exception:%s, directory:%s) exception while loading '%s': %s:", error_type, toolsdir, tool, error_message)

            if tool not in tools_mapping:
                continue

            toolinf = tools_mapping[tool]

        tool_namespace = toolinf["namespace"]
        tool_functions = [tool_namespace[f] for f in tool_namespace if f.startswith("powerbot_")]

        tools.extend(tool_functions)

    return tools

def dynload(tool, pathname, data, tools_mapping=None):
    if tools_mapping is None:
        tools_mapping = TOOLS

    if tool not in tools_mapping:
        logger.debug("(tool:%s, file:%s) loading ...", tool, pathname)

        with open(pathname) as fd:
            code = fd.read()

        tools_mapping[tool] = {
            "mtime" : os.path.getmtime(pathname),
            "code" : code,
            "namespace" : {
                "POWERBOT_DATA" : data
            }
        }

        exec(tools_mapping[tool]["code"],
             tools_mapping[tool]["namespace"],
             tools_mapping[tool]["namespace"])

    toolinf = tools_mapping[tool]

    prev_mtime = toolinf["mtime"]
    current_mtime = os.path.getmtime(pathname)

    if prev_mtime == current_mtime:
        return toolinf

    logger.debug("(tool:%s, file:%s) removing from cache", tool, pathname)

    del tools_mapping[tool]

    return dynload(tool, pathname, data, tools_mapping)

def _escape_bad_output(output):
    allowed_tags = {"b", "i", "code", "pre", "a"}

    output = nh3.clean(output,
        tags=allowed_tags,
        clean_content_tags={"think"},
        attributes={})

    index = 0
    lines = output.splitlines()

    for line in lines:
        if len(line.strip()) > 0:
            break

        index += 1

    lines = lines[index:]

    output = "\n".join(lines)

    return output

def _limit_length(text):
    chunk_size = int(os.getenv("POWERBOT_TEXT_CHUNK_SIZE"))

    return text[:chunk_size]
