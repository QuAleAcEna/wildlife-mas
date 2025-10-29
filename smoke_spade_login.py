import asyncio

import spade
from spade.agent import Agent


class SmokeAgent(Agent):
    async def setup(self):
        print(f"[OK] Logged in as {self.jid}")


async def main(args=None):
    a = SmokeAgent("sensor@localhost", "sensor123")
    await a.start(auto_register=True)  # create account on the embedded server
    await asyncio.sleep(0.5)  # let setup run before shutting everything down
    await a.stop()

if __name__ == "__main__":
    spade.run(main())  # start local XMPP for this run
