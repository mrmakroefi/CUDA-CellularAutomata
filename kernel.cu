
#include "cuda_runtime_api.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"

#include <iostream>
#include <algorithm>
#include <chrono>
#include <vector>
#include <queue>
#include <ppl.h>

#define dllexp __declspec(dllexport)
#define uint unsigned int
#define byte uint8_t
#define ushort unsigned short
#define vector std::vector

class Room {
public:
	vector<uint> tiles;
	vector<uint> edgesTiles;
	vector<Room*> connectedRooms;
	uint roomSize;
	bool isAccessibleFromMainRoom;
	bool isMainRoom;

	Room() {

	}

	Room(vector<uint> roomTiles, byte* map, uint width, uint height) {
		isAccessibleFromMainRoom = false;
		isMainRoom = false;
		tiles = roomTiles;
		roomSize = roomTiles.size();

		// get the edges tiles of the room
		for each (uint tile in tiles)
		{
			uint x = tile % width;
			uint y = (tile - x) / width;

			int xdown = x - 1, xup = x + 1;
			int ydown = y - 1, yup = y + 1;

			for (int X = xdown; X <= xup; X++) {
				for (int Y = ydown; Y <= yup; Y++) {
					if (X == x || Y == y) {
						if (map[Y * width + X] == 1) {
							edgesTiles.push_back(tile);
						}
					}
				}
			}
		}
	}

	void SetAccessibleFromMainRoom() {
		if (!isAccessibleFromMainRoom) {
			isAccessibleFromMainRoom = true;
			for (size_t i = 0; i < connectedRooms.size(); i++)
			{
				connectedRooms.at(i)->SetAccessibleFromMainRoom();
			}
		}
	}

	// connect two rooms
	static void ConnectRooms(Room *roomA, Room *roomB) {
		if (roomA->isAccessibleFromMainRoom) {
			roomB->SetAccessibleFromMainRoom();
		}
		else if (roomB->isAccessibleFromMainRoom) {
			roomA->SetAccessibleFromMainRoom();
		}
		roomA->connectedRooms.push_back(roomB);
		roomB->connectedRooms.push_back(roomA);
	}

	// check the connection between two rooms
	bool isConnected(Room *otherRoom) {
		bool found = false;
		for (size_t i = 0; i < connectedRooms.size(); i++)
		{
			if (connectedRooms.at(i)->isSame(otherRoom)) {
				found = true;
				break;
			}
		}
		return found;
	}

	bool isSame(const Room* room) {
		bool flag = this == room;
		/*printf("checking %p -> %i with %p -> %i\n", this, roomSize, &room, room->roomSize);
		if (flag) {
			printf("there's same room detected\n");
		}*/
		return flag;
	}
};

extern "C" {

	dllexp char* getDeviceName();
	dllexp void GPU_simulateCA(byte* data, ushort width, ushort height, byte chanceToAlive, byte starvationLimit, byte birthLimit, int seed, byte iterationsCount, ushort threadsCount);
	dllexp void CPU_simulateCA(byte* data, ushort width, ushort height, byte chanceToAlive, byte starvationLimit, byte birthLimit, int seed, byte iterationsCount);
	dllexp void CPU_floodFill(byte *data, uint width, uint height, uint wallThreshold, uint floorThreshold);

	dllexp void CPU_initMap(byte* data, uint size, byte chanceToAlive, uint seed);
}

vector<uint> GetRegionTiles(byte *data, uint width, uint height, byte *mapFlags, uint id);
vector<vector<uint>> GetRegions(byte *data, uint width, uint height, byte tileType);
void ConnectClosestRooms(vector<Room*> allRooms, byte* data, uint width, uint height, bool forceAccessibility = false);
void CreatePassage(byte* data, uint width, uint height, Room *roomA, Room *roomB, uint tileA, uint tileB);
void DrawCircle(uint id, int r, byte* data, uint width, uint height);
vector<uint> GetLine(uint from, uint to, uint width);

int sign(int x) {
	return (x > 0) ? 1 : ((x < 0) ? -1 : 0);
}

// init map, fill map with 0 and 1
/*__global__ void GPU_InitMap(byte* output, uint size, uint aliveChance, uint seed) {

for (uint cellId = blockIdx.x * blockDim.x + threadIdx.x; cellId < size; cellId += blockDim.x * gridDim.x) {
curandState state;
curand_init(seed, cellId, 0, &state);
uint value = (uint)(curand_uniform(&state) * 100) <= aliveChance ? 1 : 0;
output[cellId] = value;
}
}*/

__global__ void simulateCA(byte* data, uint width, uint height, byte* dataBuffer, byte starvationLimit, byte birthLimit) {
	uint worldSize = width * height;

	// initiate threads
	for (uint cellId = blockIdx.x * blockDim.x + threadIdx.x; cellId < worldSize; cellId += blockDim.x * gridDim.x) {

		// define x and y coord
		uint x = cellId % width;
		uint y = (cellId - x) / width;

		if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
			dataBuffer[cellId] = 1;
		}
		else {
			// get the neighboors
			uint aliveNeighboors = 0;
			for (int j = -1; j < 2; j++) {
				for (int i = -1; i < 2; i++) {
					// ignore the current checked cell
					if (j == 0 && i == 0) continue;

					// get neighboors cell
					int xNeighbour = x + j;
					int yNeighbour = y + i;

					// if out of map
					if (xNeighbour < 0 || yNeighbour < 0 || xNeighbour >= width || yNeighbour >= height) {
						aliveNeighboors++;
					}
					else if (data[yNeighbour * width + xNeighbour] == 1) {
						aliveNeighboors++;
					}

				}
			}

			if (data[cellId] == 1) {
				dataBuffer[cellId] = aliveNeighboors < starvationLimit ? 0 : 1;
			}
			else {
				dataBuffer[cellId] = aliveNeighboors > birthLimit ? 1 : 0;
			}
		}

	}
}

void deleteThis(int* ptr) {
	delete ptr;
}

char* getDeviceName() {

	cudaDeviceProp device;
	cudaGetDeviceProperties(&device, 0);
	char* nameLabel = new char[256];
	std::strcpy(nameLabel, device.name);
	struct cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, 0);
	std::cout << "using " << properties.multiProcessorCount << " multiprocessors" << std::endl;
	std::cout << "max threads per processor: " << properties.maxThreadsPerMultiProcessor << std::endl;
	return nameLabel;
}

void GPU_simulateCA(byte* data, ushort width, ushort height, byte chanceToAlive, byte starvationLimit, byte birthLimit, int seed, byte iterationsCount, ushort threadsCount) {
	uint worldSize = width * height;
	byte* dev_data = new byte[worldSize];
	byte* dev_dataBuffer = new byte[worldSize];

	cudaMalloc((void**)&dev_data, sizeof(byte) * worldSize);
	cudaMalloc((void**)&dev_dataBuffer, sizeof(byte) * worldSize);

	// configure thread per block
	size_t reqBlocksCount = worldSize / threadsCount;
	ushort blocksCount = (ushort)std::min((size_t)32768, reqBlocksCount);

	cudaError_t cudaStatus;

	float time;
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	CPU_initMap(data, worldSize, chanceToAlive, seed);

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Time to init world : " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0 << " ms" << std::endl;

	cudaMemcpy(dev_data, data, sizeof(byte) * worldSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_dataBuffer, data, sizeof(byte) * worldSize, cudaMemcpyHostToDevice);

	cudaEventRecord(start, 0);
	// start CA iterations
	for (byte i = 0; i < iterationsCount; i++) {
		simulateCA << <blocksCount, threadsCount >> > (dev_data, width, height, dev_dataBuffer, starvationLimit, birthLimit);

		std::swap(dev_data, dev_dataBuffer);
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Time to simulate world :   %3.1f ms \n", time);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "simulateCA kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	cudaMemcpy(data, dev_data, sizeof(byte) * worldSize, cudaMemcpyDeviceToHost);


Error:
	cudaFree(dev_data);
	cudaFree(dev_dataBuffer);
}

void CPU_initMap(byte* data, uint size, byte chanceToAlive, uint seed) {
	srand(seed);
	for (uint id = 0; id < size; id++) {
		uint value = (rand() % 100) + 1 <= chanceToAlive ? 1 : 0;
		data[id] = value;
	}
}

void CPU_simulateCA(byte* data, ushort width, ushort height, byte chanceToAlive, byte starvationLimit, byte birthLimit, int seed, byte iterationsCount) {
	uint worldSize = width * height;
	byte* dataTemp = new byte[worldSize];
	byte* dataBuffer = new byte[worldSize];

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	CPU_initMap(dataTemp, worldSize, chanceToAlive, seed);

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Time to init world : " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0 << " ms" << std::endl;

	std::memcpy(dataBuffer, dataTemp, sizeof(byte) * worldSize);

	begin = std::chrono::steady_clock::now();

	for (byte p = 0; p < iterationsCount; p++) {
		for (uint id = 0; id < worldSize; id++) {
			// define x and y coord
			uint x = id % width;
			uint y = (id - x) / width;

			if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
				dataBuffer[id] = 1;
			}
			else {
				// get the neighboors
				uint aliveNeighboors = 0;
				for (int j = -1; j < 2; j++) {
					for (int i = -1; i < 2; i++) {
						// ignore the current checked cell
						if (j == 0 && i == 0) continue;

						// get neighboors cell
						int xNeighbour = x + j;
						int yNeighbour = y + i;

						// if out of map
						if (xNeighbour < 0 || yNeighbour < 0 || xNeighbour >= width || yNeighbour >= height) {
							//printf("------ Out of map (%i, %i)\n", xNeighbour, yNeighbour);
							aliveNeighboors++;
						}
						else if (dataTemp[yNeighbour * width + xNeighbour] == 1) {
							//printf("------ Alive neighboor (%i, %i)\n", xNeighbour, yNeighbour);
							aliveNeighboors++;
						}
					}
				}

				if (dataTemp[id] == 1) {
					dataBuffer[id] = aliveNeighboors < starvationLimit ? 0 : 1;
				}
				else {
					dataBuffer[id] = aliveNeighboors > birthLimit ? 1 : 0;
				}
			}
		}

		std::swap(dataBuffer, dataTemp);
	}
	end = std::chrono::steady_clock::now();

	std::cout << "Time to simulate world : " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0 << " ms" << std::endl;

	std::memcpy(data, dataTemp, sizeof(byte) * worldSize);
	delete dataBuffer;
	delete dataTemp;
}

vector<uint> GetRegionTiles(byte *data, uint width, uint height, byte *mapFlags, uint id) {
	vector<uint> tiles;
	byte tileType = data[id];
	std::queue<uint> q;
	q.push(id);
	mapFlags[id] = 1;

	while (!q.empty()) {
		uint tile = q.front();
		q.pop();
		tiles.push_back(tile);

		uint x = tile % width,
			y = (tile - x) / width;

		int xdown = x - 1, xup = x + 1;
		int ydown = y - 1, yup = y + 1;

		for (int X = xdown; X <= xup; X++) {
			for (int Y = ydown; Y <= yup; Y++) {
				if ((X == x || Y == y)) {
					if ((X >= 0 && Y >= 0 && X < width && Y < height)) {
						uint neighbourID = (Y * width) + X;
						if (mapFlags[neighbourID] == 0 && data[neighbourID] == tileType) {
							mapFlags[neighbourID] = 1;
							q.push(neighbourID);
						}
					}
				}
			}
		}
	}

	return tiles;
}

vector<vector<uint>> GetRegions(byte *data, uint width, uint height, byte tileType) {
	uint size = width * height;
	vector<vector<uint>> regions;
	byte *mapFlags = new byte[size];
	size_t i;

	for (i = 0; i < size; i++) {
		mapFlags[i] = 0;
	}

	for (i = 0; i < size; i++) {
		if (mapFlags[i] == 0 && data[i] == tileType) {
			vector<uint> newRegion;
			newRegion = GetRegionTiles(data, width, height, mapFlags, i);
			regions.push_back(newRegion);
		}
	}

	delete mapFlags;
	return regions;
}

void CPU_floodFill(byte *data, uint width, uint height, uint wallThreshold, uint floorThreshold) {
	size_t i;
	std::chrono::steady_clock::time_point begin;
	std::chrono::steady_clock::time_point end;

	// remove small wall
	if (wallThreshold > 0) {
		begin = std::chrono::steady_clock::now();
		vector<vector<uint>> wallRegions = GetRegions(data, width, height, 1);
		for (i = 0; i < wallRegions.size(); i++)
		{
			if (wallRegions.at(i).size() < wallThreshold) {
				for (int j = 0; j < wallRegions.at(i).size(); j++)
				{
					data[wallRegions.at(i).at(j)] = 0;
				}
			}
		}
		end = std::chrono::steady_clock::now();
		std::cout << "Took " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0 << " ms" << " to remove annoying wall(s)." << std::endl;
	}

	// fill small room
	if (floorThreshold > 0) {
		begin = std::chrono::steady_clock::now();
		vector<vector<uint>> floorRegions = GetRegions(data, width, height, 0);
		vector<Room*> survivedRooms;
		for (i = 0; i < floorRegions.size(); i++)
		{
			if (floorRegions.at(i).size() < floorThreshold) {
				for (int j = 0; j < floorRegions.at(i).size(); j++)
				{
					data[floorRegions.at(i).at(j)] = 1;
				}
			}
			else {
				Room* room = new Room(floorRegions.at(i), data, width, height);
				survivedRooms.push_back(room);
			}
		}
		end = std::chrono::steady_clock::now();
		std::cout << "Took " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0 << " ms" << " to fill useless room(s)." << std::endl;

		begin = std::chrono::steady_clock::now();
		// descending order
		std::sort(survivedRooms.begin(), survivedRooms.end(), [](Room* a, Room* b) {
			return a->roomSize > b->roomSize;
		});

		survivedRooms.at(0)->isMainRoom = true;
		survivedRooms.at(0)->isAccessibleFromMainRoom = true;

		ConnectClosestRooms(survivedRooms, data, width, height);
		end = std::chrono::steady_clock::now();
		std::cout << "Took " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() / 1000.0 << " ms" << " to connect room(s)." << std::endl;

	}
}

void ConnectClosestRooms(vector<Room*> allRooms, byte* data, uint width, uint height, bool forceAccessibility) {
	vector<Room*> roomListA, roomListB;

	if (forceAccessibility) {
		for (int i = 0; i < allRooms.size(); i++) {
			bool flag = allRooms.at(i)->isAccessibleFromMainRoom;

			if (flag) {
				roomListB.push_back(allRooms.at(i));
			}
			else {
				roomListA.push_back(allRooms.at(i));
			}
		}
	}
	else {
		for (int i = 0; i < allRooms.size(); i++) {
			roomListA.push_back(allRooms.at(i));
			roomListB.push_back(allRooms.at(i));
		}
	}

	int bestDistance = 0;
	uint bestTileA, bestTileB;
	Room *bestRoomA, *bestRoomB;
	bool possibleConnectionFound = false;


	for (int i = 0; i < roomListA.size(); i++) {
		if (!forceAccessibility) {
			possibleConnectionFound = false;
			if (roomListA.at(i)->connectedRooms.size() > 0) {
				continue;
			}
		}
		for (int j = 0; j < roomListB.size(); j++) {
			if (roomListA.at(i)->isSame(roomListB.at(j)) || roomListA.at(i)->isConnected(roomListB.at(j))) {
				continue;
			}

			for (int tileA = 0; tileA < roomListA.at(i)->edgesTiles.size(); tileA+=3) {
				for (int tileB = 0; tileB < roomListB.at(j)->edgesTiles.size(); tileB+=3) {
					uint tileIdA = roomListA.at(i)->edgesTiles.at(tileA);
					uint tileIdB = roomListB.at(j)->edgesTiles.at(tileB);

					int xA = tileIdA % width;
					int yA = (tileIdA - xA) / width;
					int xB = tileIdB % width;
					int yB = (tileIdB - xB) / width;

					int distance = std::pow(xA - xB, 2) + std::pow(yA - yB, 2);
					
					if (distance < bestDistance || !possibleConnectionFound) {
						bestDistance = distance;
						possibleConnectionFound = true;
						bestTileA = tileIdA;
						bestTileB = tileIdB;
						bestRoomA = roomListA.at(i);
						bestRoomB = roomListB.at(j);
					}
				}
			}
		}
		if (possibleConnectionFound && !forceAccessibility) {
			CreatePassage(data, width, height, bestRoomA, bestRoomB, bestTileA, bestTileB);
		}
	}

	if (possibleConnectionFound && forceAccessibility) {
		CreatePassage(data, width, height, bestRoomA, bestRoomB, bestTileA, bestTileB);
		ConnectClosestRooms(allRooms, data, width, height, true);
	}

	if (!forceAccessibility) {
		ConnectClosestRooms(allRooms, data, width, height, true);
	}
}

void CreatePassage(byte* data, uint width, uint height, Room *roomA, Room *roomB, uint tileA, uint tileB) {
	Room::ConnectRooms(roomA, roomB);

	vector<uint> line = GetLine(tileA, tileB, width);
	for each(uint c in line) {
		DrawCircle(c, 1, data, width, height);
	}
}

void DrawCircle(uint id, int r, byte* data, uint width, uint height) {
	for (int x = -r; x <= r; x++) {
		for (int y = -r; y <= r; y++) {
			if (x*x + y*y <= r*r) {
				int drawX = (id % width) + x;
				int drawY = ((id - id%width) / width) + y;

				if (drawX >= 0 && drawY >= 0 && drawX < width && drawY < height) {
					data[drawY * width + drawX] = 0;
				}
			}
		}
	}
}

vector<uint> GetLine(uint from, uint to, uint width) {
	vector<uint> line;

	uint x = from % width,
		y = (from - x) / width;
	uint xTo = to % width,
		yTo = (to - xTo) / width;

	bool inverted = false;
	int dx = xTo - x;
	int dy = yTo - y;

	int step = sign(dx);
	int gradientStep = sign(dy);

	int longest = abs(dx);
	int shortest = abs(dy);

	if (longest < shortest) {
		inverted = true;
		longest = abs(dy);
		shortest = abs(dx);

		step = sign(dy);
		gradientStep = sign(dx);
	}

	int gradientAccumulation = longest / 2;
	for (int i = 0; i < longest; i++) {
		line.push_back(y * width + x);
		gradientAccumulation += shortest;

		if (inverted) {
			y += step;
		}
		else {
			x += step;
		}

		if (gradientAccumulation >= longest) {
			if (inverted) {
				x += gradientStep;
			}
			else {
				y += gradientStep;
			}
			gradientAccumulation -= longest;
		}
	}
	return line;
}
