// The Win32 calls the extension makes, declared here with their own types
// rather than through windows.h, whose macros (FAR, near, far, min, max and
// hundreds more) would reach every source that includes it: `static` refuses
// windows.h anywhere under src/. Windows only, and included only there.
#pragma once

#if defined(_WIN64)
using TrimkitWin32Size = unsigned long long;  // SIZE_T
#else
using TrimkitWin32Size = unsigned long;  // SIZE_T
#endif
// what GetProcAddress answers (FARPROC), cast to the function it names
using TrimkitWin32Procedure = void(__stdcall*)();

extern "C" {
// the pool sleeping and waking on a word
__declspec(dllimport) int __stdcall WaitOnAddress(volatile void* address, void* expected, TrimkitWin32Size size,
                                                  unsigned long timeout_ms);
__declspec(dllimport) void __stdcall WakeByAddressAll(void* address);
__declspec(dllimport) int __stdcall CloseHandle(void* handle);
// the memory still available (src/trimkit.cpp): `status` is a MEMORYSTATUSEX
__declspec(dllimport) int __stdcall GlobalMemoryStatusEx(void* status);
// the Vulkan loader, opened by name (src/vulkan/vulkan.hpp)
__declspec(dllimport) void* __stdcall LoadLibraryA(const char* name);
__declspec(dllimport) TrimkitWin32Procedure __stdcall GetProcAddress(void* library, const char* name);
}
