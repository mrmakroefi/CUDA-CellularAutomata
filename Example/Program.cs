using System;
using System.Runtime.InteropServices;

namespace CUDACheck {
    class Program {
        static void Main(string[] args)
        {
            IntPtr ptrName;
            string devName;

            ptrName = CUDA.getDeviceName();
            devName = Marshal.PtrToStringAnsi(ptrName);
            Console.WriteLine(devName);

            byte iterations = 2;
            ushort width = 512;
            ushort height = 512;
            uint length = (uint)(width * height);

            Console.WriteLine("\n====== World Information ======");
            Console.WriteLine("- World size: (" + width + "," + height + ")");
            Console.WriteLine("- Total cell: " + length);
            Console.WriteLine("===============================\n");

            byte[] roll = new byte[length];
            IntPtr testNum = Marshal.AllocHGlobal(roll.Length);
            Marshal.Copy(roll, 0, testNum, roll.Length);
            
            //CUDA.CPU_simulateCA(testNum, width, height, 35, 3, 3, 1, iterations);
            CUDA.GPU_simulateCA(testNum, width, height, 35, 3, 3, 1, iterations, 128);

            CUDA.CPU_floodFill(testNum, width, height, 12, 8);

            Marshal.Copy(testNum, roll, 0, roll.Length);
            
           // PrintData(roll, width, length);


            Console.ReadKey();

            Marshal.FreeHGlobal(testNum);
            Marshal.FreeHGlobal(ptrName);
        }

        static void PrintData(byte[] data, ushort width, uint worldSize)
        {
            for (int i = 0; i < worldSize; i++)
            {
                Console.Write(data[i] + "");
                if ((i + 1) % width == 0 && i != 0)
                {
                    Console.WriteLine();
                }
            }
        }
    }
}
