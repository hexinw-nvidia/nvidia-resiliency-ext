FT Launcher & Inprocess integration
***********************************

.. warning::

   **InJob+InProcess integration is not supported.** The FT launcher's ``any-failed`` restart policy is incompatible with inprocess recovery mechanisms.

The FT launcher uses the ``any-failed`` restart policy, which conflicts with inprocess recovery in several ways:

1. **Conflicting restart logic**: The FT launcher restarts all workers when any worker fails, while inprocess recovery attempts to recover individual ranks.

2. **Upscaling conflicts**: FT launcher automatically enables upscaling, which can restart training when new nodes become available, conflicting with inprocess recovery.

3. **Undefined behavior**: The combination of FT launcher restarts and inprocess recovery can lead to race conditions and unpredictable behavior.

Recommendations
===============

Use one of the following approaches instead:

1. **FT launcher only**: Use the FT launcher's fault tolerance without inprocess recovery.

2. **Inprocess only**: Use inprocess recovery without the FT launcher.

.. note::

   The ``--ft-restart-policy`` argument is deprecated and will be removed in a future release. The FT launcher now only supports the ``any-failed`` restart policy.
