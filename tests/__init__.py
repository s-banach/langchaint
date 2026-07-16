"""Offline tests: constructed SDK objects and fake providers through the package.

Every test runs without network access or API keys.
Adapter tests feed constructed anthropic and openai SDK objects into the module-level helpers
and inspect the normalized result;
client tests drive BoundLLM and StreamHandle with fake BoundProvider and ProviderStream implementations.
"""
