//
// Created by chips on 03.02.19.
//

#ifndef CHIPSYOLOV3_LIBTORCH_SPLITTER_HPP
#define CHIPSYOLOV3_LIBTORCH_SPLITTER_HPP

#include "Trimmer.hpp"

/**
 * The splitter is a simple utility class which is able to split std::strings into a vector
 * of std::string. Only used to provide an elegant way of splitting, and to eliminate whitespace problems
 * in the default implementation of splitting.
 */
class Splitter {
private:
    Trimmer _trimmer;
public:
    inline int split(const string& str, std::vector<string>& ret_, string sep = ",")
    {
        if (str.empty())
        {
            return 0;
        }

        string tmp;
        string::size_type pos_begin = str.find_first_not_of(sep);
        string::size_type comma_pos = 0;

        while (pos_begin != string::npos)
        {
            comma_pos = str.find(sep, pos_begin);
            if (comma_pos != string::npos)
            {
                tmp = str.substr(pos_begin, comma_pos - pos_begin);
                pos_begin = comma_pos + sep.length();
            }
            else
            {
                tmp = str.substr(pos_begin);
                pos_begin = comma_pos;
            }

            if (!tmp.empty())
            {
                _trimmer.trim(tmp);
                ret_.push_back(tmp);
                tmp.clear();
            }
        }
        return 0;
    }

    inline int split(const string& str, std::vector<int>& ret_, string sep = ",")
    {
        std::vector<string> tmp;
        split(str, tmp, sep);

        for(int i = 0; i < tmp.size(); i++)
        {
            ret_.push_back(std::stoi(tmp[i]));
        }
    }

};

#endif //CHIPSYOLOV3_LIBTORCH_SPLITTER_HPP
