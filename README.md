# Powerbot

**Powerbot** is an experimental, opinionated and potentially powerfull AI-powered Telegram assistant, supporting both Gemini and any OpenAI-compatible APIs. Thanks to **hooks**, **tools**, and integration with **MCP**, Powerbot can be easily extended. LLM settings, such as **temperature**, **top_p**, **top_k**, **max_tokens**, **frequency_penalty**, **presence_penalty**, and even **system instruction**, are easily configurable, so you can customize the "personality" to your liking. The history is saved via **lmdb** for faster reading (using mmap). Through this bot, you can analyze documents, photos, audio, videos, voice, stickers, video notes, and animations. Additionally, there are two ways to use this bot: private use (authorized users only) and public use (anyone).

## Configuring

Configuration is done using environment variables, and a dotenv file is supported. There is a very descriptive example in [examples/default.env](examples/default.env). Please note that all environment variables are required, except for those marked as commented out.

## Deploying

You can use the standard `uv` commands to install this project; just remember to also install `ffmpeg` and `textract` (which can be installed using `uv` as well) if you plan to use OpenAI instead of Gemini, or, to make things easier, use the [Director](https://github.com/DtxdF/director) file from this repository.

```sh
git clone https://github.com/DtxdF/powerbot.git
cd ./powerbot/
cp examples/default.env .env
$EDITOR .env
appjail-director up
```

**Note**: Throughout the rest of this document, it is assumed that you are running Powerbot within a jail.

## Tools

Once Powerbot is up and running, it can transcribe audio, analyze videos and photos, or simply chat. However, you may want to add more features, and that’s what "Tools" (formerly known as "Function Calls") are for. This is the low-level part of MCPs, but in Powerbot you can create Python code that is dynamically loaded at runtime. The LLM model determines which function to call based on the arguments, type hints, and the docstring. For example, a simple tool is to retrieve the user ID on Telegram. The only requisite is that you prefix your functions with `powerbot_`.

**/var/appjail-volumes/powerbot/data/tools/user/user_info**:

```python
def powerbot_get_userid(ignore: str = "") -> int:
    """
    Get user ID in Telegram.
    """

    message = POWERBOT_DATA["AIOGRAM_MESSAGE"]
    userid = message.from_user.id

    return userid
```

**Note**: Attentive readers may have noticed that there is a parameter called `ignore` that isn't used. This is a workaround for the OpenAI Agents SDK and isn't required in the Google GenAI SDK.

Powerbot detects changes based on the modification date (mtime). If the modification date has changed, the code is reloaded, otherwise, it is loaded from memory. The code already has a namespace defined for both global and local variables (see below), so you can access to some useful variables.

Now, in the chat where the bot is running, you can type `Get my ID in telegram` and it will respond with something like `Your Telegram user ID is 1234567890`.

### Hooks

Hooks work the same way as tools: they are loaded at runtime and reloaded if their mtime changes. However, hooks must be asynchronous functions and must accept at least one argument. This argument is the result of the last (succesfully) function called. This approach creates a pipeline structure, but using Python code. Additionally, keep in mind that functions are sorted by name, and if the last function returns a value other than `None`, Powerbot will do nothing (e.g.: it will not make any calls to the LLM model or respond to any messages).

**/var/data/hooks/user/allow_messages**:

```python
async def powerbot_deny_format(last):
    message = POWERBOT_DATA["AIOGRAM_MESSAGE"]

    if message.photo is not None \
            or message.document is not None:
        await message.answer("Unsupported format.")
        # Anything (except None) will just works!
        return False
```

### Namespaces

As you've probably noticed, there's a variable called `POWERBOT_DATA` that tools and hooks can use to access other useful variables, such as the LLM client instance or the Telegram message.

* `AIOGRAM_MESSAGE`: Message instance. See also https://docs.aiogram.dev/en/latest/api/types/message.html
* `AIOGRAM_BOT`: Bot instance. See also https://docs.aiogram.dev/en/latest/api/bot.html
* `GEMINI_CLIENT`: `google.genai.Client(...)` instance
* `GEMINI_CHAT`: `client.aio.chats.create(...)` instance.
* `OPENAI_CLIENT`: `openai.AsyncOpenAI(...)` instance.

## MCP

MCP is configured in [Hjson](https://hjson.github.io/) format. As with tools and hooks, the configuration is loaded once at runtime and reloaded if the mtime changes. Below is a simple example that uses both `uvx` and `npx`:

```json
{
    "mcpTimeouts" : {
        // Gemini-only
        "timeout" : 300,
        // OpenAI-only
        "connect_timeout_seconds" : 300,
        "cleanup_timeout_seconds" : 300
    },
    "mcpServers" : {
        "filesystem" : {
            "command" : "npx",
            "args" : [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                "/data/mcp-server-filesystem"
            ],
            "env" : {
                "npm_config_cache" : "/cache/npm"
            },
            "enabled" : true, // optional. Default is true.
            "timeout" : 300 // OpenAI-only
        },
        "fetch" : {
            "command" : "uvx",
            "args" : ["mcp-server-fetch"],
            "timeout" : 300,
            "env" : {
                "UV_CACHE_DIR" : "/cache/uv"
            }
        }
    }
}
```

All timeouts are set to `10` by default, but this isn't realistic if you're running `uvx` or `npx` in a fresh environment. Another approach, if you want to keep the timeouts low, is to run manually first and shut down the MCP server (with CTRL-C) once everything has been cached.

## Notes

* The key used to store the history in the LMDB database is created using the syntax `history_<provider>_<user-type>_<user-id>`, where `<provider>` is `gemini` or `openai`, `<user-type>` is `manager` or `user` (depending on `POWERBOT_IS_MANAGER`) and `<user-id>` is the user ID on Telegram. This means that the history isn't shared between providers or between different instance types.
