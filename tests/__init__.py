"""Offline tests: constructed SDK objects and fake adapters through langchaint.

Every test runs without network access or API keys.
Adapter tests feed constructed anthropic and openai SDK objects into the module-level helpers
and inspect the normalized result;
client tests drive BoundLLM and StreamHandle with fake BoundAdapter and AdapterStream implementations.
"""
