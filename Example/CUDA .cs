using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Runtime.InteropServices;

namespace CUDACheck {
    public static class CUDA {
        [DllImport("DLL_CUDA")]
        public static extern IntPtr getDeviceName();
        
        [DllImport("DLL_CUDA")]
        public static extern void GPU_simulateCA(IntPtr data, ushort width, ushort height,byte chanceToAlive, byte starvationLimit, byte birthLimit, int seed, byte iterationsCount, ushort threadsCount);

        [DllImport("DLL_CUDA")]
        public static extern void CPU_initMap(IntPtr data, uint size, byte chanceToAlive, uint seed);

        [DllImport("DLL_CUDA")]
        public static extern IntPtr CPU_simulateCA(IntPtr input, ushort width, ushort height, byte chanceToAlive, byte starvationLimit, byte birthLimit, int seed, byte iterationsCount);
        
        [DllImport("DLL_CUDA")]
        public static extern void CPU_floodFill(IntPtr data, uint width, uint height, uint wallThreshold, uint floorThreshold);
    }
}
