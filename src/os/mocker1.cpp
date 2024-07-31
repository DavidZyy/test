#include <cstdlib>
#include <string>
#include <cstring>
#include <sys/types.h>
#include <vector>
#include <iostream>
#include <sched.h>

#include <unistd.h>
#include <sys/wait.h>
#include <sys/mount.h>

using std::cout;
using std::endl;
using std::cerr;
using std::string;

static void run(int argc, char **argv);
static string cmd(int argc, char **argv);
static void run_child(int argc, char **argv);

const char *child_hostname = "container";

int main(int argc, char **argv) {
    if (argc < 3) {
        cerr << "Too few arguments" << endl;
        exit(-1);
    }

    if (!strcmp(argv[1], "run")) {
        run(argc - 2, &argv[2]);
    }
}

// static void run(int argc, char **argv) {
//     cout << "Running " << cmd(argc, argv) << endl;
//     execvp(argv[0], argv);
// }

static void run(int argc, char **argv) {
    cout << "Parent running " << cmd(argc, argv) << " as " << getpid() << endl;

    if (unshare(CLONE_NEWPID)) {
        cerr << "Failed to unshare PID namespace: " << strerror(errno) << endl;
        exit(-1);
    }

    pid_t child_pid = fork();

    if (child_pid < 0) {
        cerr << "Failed to fork" << endl;
        return;
    }

    if (child_pid) {
        if (waitpid(child_pid, NULL, 0) < 0) {
            cerr << "Failed to wait for child" << endl;
        } else {
            cout << "Child exited" << endl;
        }
    } else {
        run_child(argc, argv);
    }

}

static void run_child(int argc, char **argv) {
    cout << "Child running " << cmd(argc, argv) << " as " << getpid() << endl;

    int flags = CLONE_NEWUTS;

    if (unshare(flags) < 0) {
        cerr << "Failed to unshare: " << strerror(errno) << endl;
        exit(-1);
    }

    // if (chroot("/home/zhuyangyang/project/") < 0) {
    //     cerr << "Failed to chroot: " << strerror(errno) << endl;
    //     exit(-1);
    // }

    // if (chdir("/") < 0) {
    //     cerr << "Failed to chdir: " << strerror(errno) << endl;
    //     exit(-1);
    // }

    if (sethostname(child_hostname, strlen(child_hostname)) < 0) {
        cerr << "Failed to set hostname: " << strerror(errno) << endl;
        exit(-1);
    }

    if (execvp(argv[0], argv)) {
        cerr << "Failed to exec: " << strerror(errno) << endl;
    }
}

static string cmd(int argc, char **argv) {
    string cmd = "";
    for (int i = 0; i < argc; i++) {
        cmd.append(argv[i] + string(" "));
    }
    return cmd;
}

