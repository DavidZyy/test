#include<string>
#include<iostream>
#include<cmath>
#include<iomanip>
#include<cstdio>

using namespace std;

int main() {
    char letter1, letter2;
    double value1, value2;
    double T, D, H;
    double e, h;

    while (true) {
        // Read a line of input
        string line;
        getline(cin, line);

        // Check for the end of input
        if (line == "E") {
            break; // Exit the loop when "E" is encountered
        }

        // Parse the input line
        int items_read = sscanf(line.c_str(), " %c %lf %c %lf", &letter1, &value1, &letter2, &value2);

        if(letter1 == 'T' && letter2 == 'D' ||
            letter1 == 'D' && letter2 == 'T') {
            if(letter1 == 'T') {
                T = value1;
                D = value2;
            } else {
                T = value2;
                D = value1;
            }
            e = 6.11 * exp(5417.7530 * ((1.0 / 273.16) - (1.0 / (D+ 273.16))));
            h = (0.5555)*(e - 10.0);
            H = T + h;
        } else if (letter1 == 'T' && letter2 == 'H' || 
            letter1 == 'H' && letter2 == 'T') {
            if(letter1 == 'T') {
                T = value1;
                H = value2;
            } else {
                T = value2;
                H = value1;
            }
            h = H - T;
            e = h / 0.5555 + 10;
            D = 1.0 / ((1.0/273.16) - log(e/6.11)/5417.7530) - 273.16;
        } else { // D H
            if(letter1 == 'D') {
                D = value1;
                H = value2;
            } else {
                D = value2;
                H = value1;
            }
            e = 6.11 * exp(5417.7530 * ((1.0 / 273.16) - (1.0 / (D+ 273.16))));
            h = (0.5555)*(e - 10.0);
            T = H - h;
        }


        cout << "T "  << fixed << setprecision(1) << T
             << " D " << fixed << setprecision(1) << D
             << " H " << fixed << setprecision(1) << H << endl;
    }

    return 0;
}