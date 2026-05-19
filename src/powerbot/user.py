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

import os

import aiogram

import powerbot.ai

DISPATCHER = aiogram.Dispatcher()

@DISPATCHER.message(aiogram.filters.CommandStart())
async def handle_start_message(message, bot):
    start_prompt = os.getenv("POWERBOT_USER_START_PROMPT")

    if start_prompt is not None:
        start_prompt = f"[POWERBOT: start]: {start_prompt}"

        await powerbot.ai.make_response(message, 
            bot=bot, text=start_prompt)

@DISPATCHER.message()
async def handle_message(message, bot):
    await powerbot.ai.make_response(message, bot=bot)

def make_bot():
    bot = aiogram.Bot(
        token=os.getenv("POWERBOT_USER_TOKEN"),
        default=aiogram.client.default.DefaultBotProperties(
            parse_mode=aiogram.enums.ParseMode.HTML
        )
    )

    return bot

async def start_polling():
    bot = make_bot()

    await DISPATCHER.start_polling(bot)
