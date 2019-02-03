//
// Created by chips on 02.02.19.
//
#ifndef CHIPSYOLOV3_LIBTORCH_CONFIGLOADER_HPP
#define CHIPSYOLOV3_LIBTORCH_CONFIGLOADER_HPP

#include <torch/torch.h>
#include "Splitter.hpp"
#include "Trimmer.hpp"

using Config = std::vector<std::map<std::string, std::string>>;

class ConfigLoader {
private:
    Splitter _splitter;
public:
    ConfigLoader()
        : _splitter() {}
    /**
     *  Reads a darknet-formatted config file and extracts the structural information out it.
     *
     * @param cfgFile - filename of the configuration file.
     * @return vector containing the information about each layer. the order in the vector is the same as the order
     *          in the desired network.
     */
    std::vector<std::map<std::string, std::string>> loadFromConfig(const std::string& cfgFile) {
        std::ifstream fs(cfgFile);
        std::string line;

        std::vector<std::map<std::string, std::string>> blocks;

        if(!fs)
        {
            std::string error = "Fail to load cfg file:" + cfgFile;
            throw std::runtime_error(error);
        }

        while (getline (fs, line))
        {
            Trimmer::trim(line);

            if (line.empty())
            {
                continue;
            }

            if ( line.substr (0,1)  == "[")
            {
                std::map<string, string> block;

                std::string key = line.substr(1, line.length() -2);
                block["type"] = key;

                blocks.push_back(block);
            }
            else
            {
                std::map<string, string> *block = &blocks[blocks.size() -1];

                std::vector<string> op_info;

                _splitter.split(line, op_info, "=");

                if (op_info.size() == 2)
                {
                    string p_key = op_info[0];
                    string p_value = op_info[1];
                    block->operator[](p_key) = p_value;
                }
            }
        }
        fs.close();

        return blocks;
    }

};

#endif //CHIPSYOLOV3_LIBTORCH_CONFIGLOADER_HPP
