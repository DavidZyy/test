#include <string>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <climits>
#include <set>
#include <queue>
#include <cstring>
#include <queue>

// using namespace std;

class inst {
private:
    char op[10];
    int rd, rs1, rs2;

public:
    inst(const char* operation, int destination, int source1, int source2)
        : rd(destination), rs1(source1), rs2(source2) {
        strncpy(op, operation, sizeof(op) - 1);
        op[sizeof(op) - 1] = '\0'; // Ensure null-terminated string
    }

    const char* getOperation() const { return op; }
    int getDestination() const { return rd; }
    int getSource1() const { return rs1; }
    int getSource2() const { return rs2; }
    ~inst() {}
};

class ReservationStation {
public:
    ReservationStation() {
        busy = false;
        op = "";
        vj = 0;
        vk = 0;
        qj = 0;
        qk = 0;
        result = 0;
        resultStation = 0;
    }

    void SetOperation(const std::string& operation) { op = operation; }
    void SetBusy(bool isBusy) { busy = isBusy; }
    void SetVj(int value) { vj = value; }
    void SetVk(int value) { vk = value; }
    void SetQj(int station) { qj = station; }
    void SetQk(int station) { qk = station; }
    void SetResult(int value, int station) {
        result = value;
        resultStation = station;
    }
    bool IsBusy() const { return busy; }
    std::string GetOperation() const { return op; }
    int GetVj() const { return vj; }
    int GetVk() const { return vk; }
    int GetQj() const { return qj; }
    int GetQk() const { return qk; }
    int GetResult() const { return result; }
    int GetResultStation() const { return resultStation; }

protected:
    bool busy;
    std::string op;
    int vj;
    int vk;
    int qj;
    int qk;
    int result;
    int resultStation;
};

class AddReservationStation : public ReservationStation {
public:
    AddReservationStation() {
        SetOperation("ADD");
    }
};

class MultiplyReservationStation : public ReservationStation {
public:
    MultiplyReservationStation() {
        SetOperation("MUL");
    }
};

queue<inst> inst_q;

int main() {
    // Example usage:
    AddReservationStation addRS[3];
    MultiplyReservationStation mulRS[3];

    return 0;
}



