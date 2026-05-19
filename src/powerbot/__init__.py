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
import logging
import os
import sys

from dotenv import load_dotenv

async def start():
    dotenv = os.getenv("POWERBOT_DOTENV", ".env")

    if os.path.isfile(dotenv):
        with open(dotenv) as fd:
            load_dotenv(stream=fd)

    log_level = os.getenv("POWERBOT_LOG_LEVEL", "")
    log_level = log_level.upper()

    if log_level == "CRITICAL":
        log_level = logging.CRITICAL

    elif log_level == "ERROR":
        log_level = logging.ERROR

    elif log_level == "WARNING":
        log_level = logging.WARNING

    elif log_level == "INFO":
        log_level = logging.INFO

    elif log_level == "DEBUG":
        log_level = logging.DEBUG

    else:
        log_level = logging.NOTSET

    logging.basicConfig(level=log_level, stream=sys.stdout)

    is_manager = os.getenv("POWERBOT_IS_MANAGER", "0")

    if is_manager != "0":
        import powerbot.manager

        await powerbot.manager.start_polling()

    else:
        import powerbot.user

        await powerbot.user.start_polling()

def main():
    return asyncio.run(start())
