#include "GameSim.h"
#include "ClassicalTeam.h"
#include <logging-utils-lib/filesystem.h>
#include <logging-utils-lib/yaml.h>
#include <logging-utils-lib/logging.h>
#include <progress_bar.h>
#include <Eigen/Core>

int main(int argc, char* argv[])
{
    std::vector<std::string> program_args;
    if (argc > 1)
        program_args.assign(argv + 1, argv + argc);
    else
        return 1;

    // Progress bar
    ProgressBar pb;

    // Read config file values
    std::string configfname = program_args[0];
    std::string logdirname = filesys::dirName(configfname) + "/logs";
    filesys::createDirIfNotExist(logdirname);
    bool log_data;
    std::string logname, logfile;
    int winning_score;
    double dt, t_max;
    logging::get_yaml_node<bool>("log_data", configfname, log_data);
    logging::get_yaml_node<std::string>("log_name", configfname, logname);
    logfile = logdirname + "/" + logname + ".log";
    if (filesys::file_exists(logfile))
        std::remove(logfile.c_str());
    logging::get_yaml_node<int>("winning_score", configfname, winning_score);
    logging::get_yaml_node<double>("dt", configfname, dt);
    logging::get_yaml_node<double>("t_max", configfname, t_max);
    Eigen::Vector4d x0_ball;
    logging::get_yaml_eigen("x0_ball", configfname, x0_ball);

    // Game sim
    GameSim sim;
    sim.reset(false, dt, winning_score, x0_ball, log_data, logfile);

    // Game loop
    int N = static_cast<int>(t_max / dt);
    pb.init(static_cast<int>(N), 50);
    int counter = 0;
    pb.print(counter);
    while (counter < N && sim.undecided())
    {
        sim.run();
        pb.print(counter);
        counter++;
    }

    return 0;
}
