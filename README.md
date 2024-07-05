# Taichi-UnityExample (exploring MPM)

## Common problems during set up
"InvalidOperationException: Ignored launch because kernel handle is null"
this is because taichi_c_api.dll and taichi_unity.dll in Assets/Plugins/X86_64 are incompatible with the AOT modules in Assets/Resources/TaichiModules
- Solution: copy the correct taichi_c_api.dll from your pip-installed taichi to Assets/Plugins/X86_64,then rebuild the AOT modules in Assets/Resources/TaichiModules using same version of taichi

## How to stream to quest3?
-install meta quest link
-connect to PC，then open quest link on VR headset
-open unity project, set open-XR plugin
-click “play”, then you can see the scene in VR headset