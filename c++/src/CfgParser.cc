#include "CfgParser.h"
#include <fstream>

using namespace std;
TorchUtils::CfgParser::CfgParser()
{
    lSecBlockSymb_ = "["; // regexp style
    rSecBlockSymb_ = "]"; // regexp style
    optAssignSymb_ = "=";
    optListSepSymb_ = ",";
    commentSymb_ = "#";
}

TorchUtils::CfgParser::CfgParser(string filename) : CfgParser()
{
    init(filename);
}

TorchUtils::CfgParser::~CfgParser()
{
}

bool TorchUtils::CfgParser::init(string filename)
{
    ifstream cfgfile(filename);
    if (!cfgfile.is_open())
    {
        cerr << "** TorchUtils::CfgParser::init error: file " << filename << " not found" << endl;
        return false;
    }
    filename_ = filename;

    string line;
    string currsect = "";
    long linenum = 0;

    while (getline(cfgfile, line))
    {
        ++linenum;
        trimLine(line);                                    // remove whitespaces
        line = line.substr(0, line.find(commentSymb_, 0)); // remove comments
        if (line.empty())
            continue;
        // search for new block declaration
        auto lblock = line.find(lSecBlockSymb_);
        auto rblock = line.find(rSecBlockSymb_);
        auto assign = line.find(optAssignSymb_);

        // option block
        if (lblock != string::npos && rblock != string::npos && (rblock - lblock) > 0 && assign == string::npos)
        {
            string secname = line.substr(lblock + 1, rblock - lblock - 1);
            if (config_.find(secname) != config_.end())
            {
                cerr << "** TorchUtils::CfgParser:: fatal: section " << secname << " multiply defined, cannot parse file" << endl;
                return false;
            }
            currsect = secname;
            config_[secname] = optionBlock();
        }

        // line block
        else if (assign != string::npos)
        {
            if (currsect.empty())
            {
                cerr << "** TorchUtils::CfgParser:: section not defined for option on line " << linenum << endl;
                return false;
            }
            pair<string, string> p = splitOptionLine(line);
            string optName = getTrimmedLine(p.first);
            string optVal = getTrimmedLine(p.second);
            if (config_[currsect].find(optName) != config_[currsect].end())
            {
                cerr << "** TorchUtils::CfgParser:: fatal: option " << optName << " multiply defined in section " << currsect << " cannot parse file" << endl;
                return false;
            }
            config_[currsect][optName] = optVal;
        }

        // unrecognized line type
        else
        {
            cerr << "** TorchUtils::CfgParser:: cannot parse line " << linenum << endl;
            return false;
        }
    }
    cfgfile.close();
    return true;
}

bool TorchUtils::CfgParser::extend(CfgParser &cfg)
{
    // consistency check: no common section key must exist
    for (auto elem : cfg.getCfg())
    {
        if (config_.find(elem.first) != config_.end())
        {
            cerr << "** TorchUtils::CfgParser::extend : error : key " << elem.first << " double defined" << endl;
            return false;
        }
        config_[elem.first] = elem.second;
    }
    return true;
}

void TorchUtils::CfgParser::trimLine(string &line)
{
    if (line.empty())
        return;
    line = getTrimmedLine(line);
    return;
}

string TorchUtils::CfgParser::getTrimmedLine(const string &line)
{
    if (line.empty())
        return line;
    size_t first = line.find_first_not_of(" \t");
    size_t last = line.find_last_not_of(" \t");
    string res = (first != string::npos ? line.substr(first, (last - first + 1)) : ""); // if all whitespaces, return empty string
    return res;
}

pair<string, string> TorchUtils::CfgParser::splitOptionLine(string line)
{
    auto delim = line.find(optAssignSymb_);
    string optName = line.substr(0, delim);
    string optVal = line.substr(delim + optAssignSymb_.length(), string::npos);
    return make_pair(optName, optVal);
}

pair<string, string> TorchUtils::CfgParser::splitCompact(string compact)
{
    string sep = "::";
    auto delim = compact.find(sep);
    if (delim == string::npos)
    {
        cerr << "** Cfg Parser : error : cannot split compact option name " << compact << endl;
        return make_pair(string(""), string(""));
    }
    string secName = compact.substr(0, delim);
    string valName = compact.substr(delim + sep.length(), string::npos);
    return make_pair(getTrimmedLine(secName), getTrimmedLine(valName));
}

std::vector<string> TorchUtils::CfgParser::splitStringInList(std::string line)
{
    vector<string> result;
    if (endsWith(line, optListSepSymb_))
    {
        size_t lastindex = line.find_last_of(optListSepSymb_);
        line = line.substr(0, lastindex);
        trimLine(line);
    }

    size_t pos = 0;
    std::string token;
    while ((pos = line.find(optListSepSymb_)) != std::string::npos)
    {
        token = line.substr(0, pos);
        result.push_back(getTrimmedLine(token));
        line.erase(0, pos + optListSepSymb_.length());
    }
    result.push_back(getTrimmedLine(line));
    return result;
}

bool TorchUtils::CfgParser::endsWith(string line, string suffix)
{
    return line.size() >= suffix.size() &&
           line.compare(line.size() - suffix.size(), suffix.size(), suffix) == 0;
}

string TorchUtils::CfgParser::readStringOpt(string section, string option)
{
    // if (!hasOpt (section, option))
    // {
    //     cerr << "** CfgParser: option " << section << "::" << option << " not defined" << endl;
    //     return string("");
    // }
    if (!hasOpt(section, option))
    {
        throw std::runtime_error("option " + section + "::" + option + " is requested");
    }
    return config_[section][option];
}

string TorchUtils::CfgParser::readStringOpt(string compact)
{
    auto split = splitCompact(compact);
    return readStringOpt(split.first, split.second);
}

int TorchUtils::CfgParser::readIntOpt(string section, string option)
{
    string s = readStringOpt(section, option);
    return stoi(s);
}

int TorchUtils::CfgParser::readIntOpt(string compact)
{
    auto split = splitCompact(compact);
    return readIntOpt(split.first, split.second);
}

bool TorchUtils::CfgParser::readBoolOpt(string section, string option)
{
    string s = readStringOpt(section, option);
    // return stoi(s);
    if (s == "true" || s == "True")
        return true;
    else if (s == "false" || s == "False")
        return false;
    cerr << "** CfgParser: could not parse boolean option : " << s << endl;
    return false;
}

bool TorchUtils::CfgParser::readBoolOpt(string compact)
{
    auto split = splitCompact(compact);
    return readBoolOpt(split.first, split.second);
}

float TorchUtils::CfgParser::readFloatOpt(string section, string option)
{
    string s = readStringOpt(section, option);
    return stof(s);
}

float TorchUtils::CfgParser::readFloatOpt(string compact)
{
    auto split = splitCompact(compact);
    return readFloatOpt(split.first, split.second);
}

double TorchUtils::CfgParser::readDoubleOpt(string section, string option)
{
    string s = readStringOpt(section, option);
    return stod(s);
}

double TorchUtils::CfgParser::readDoubleOpt(string compact)
{
    auto split = splitCompact(compact);
    return readDoubleOpt(split.first, split.second);
}

vector<string> TorchUtils::CfgParser::readStringListOpt(string section, string option)
{
    //  if(!hasOpt(section,option)){
    //     throw std::runtime_error("option " + section + "::" + option + " is requested by the OfflineProducerHelper");
    // }
    string s = readStringOpt(section, option);
    vector<string> values = splitStringInList(s);
    return values;
}

vector<string> TorchUtils::CfgParser::readStringListOpt(string compact)
{
    auto split = splitCompact(compact);
    return readStringListOpt(split.first, split.second);
}

vector<int> TorchUtils::CfgParser::readIntListOpt(string section, string option)
{

    vector<string> vs = readStringListOpt(section, option);
    vector<int> result;
    for (string s : vs)
        result.push_back(stoi(getTrimmedLine(s)));
    return result;
}

vector<int> TorchUtils::CfgParser::readIntListOpt(string compact)
{
    auto split = splitCompact(compact);
    return readIntListOpt(split.first, split.second);
}

vector<bool> TorchUtils::CfgParser::readBoolListOpt(string section, string option)
{
    vector<string> vs = readStringListOpt(section, option);
    vector<bool> result;
    for (string s : vs)
    {
        if (s == "true" || s == "True")
            result.push_back(true);
        else if (s == "false" || s == "False")
            result.push_back(false);
        else
        {
            cerr << "CfgParser : can't parse bool list option containing: " << s << endl;
            result.push_back(false);
        }
    }
    return result;
}

vector<bool> TorchUtils::CfgParser::readBoolListOpt(string compact)
{
    auto split = splitCompact(compact);
    return readBoolListOpt(split.first, split.second);
}

vector<float> TorchUtils::CfgParser::readFloatListOpt(string section, string option)
{
    vector<string> vs = readStringListOpt(section, option);
    vector<float> result;
    for (string s : vs)
        result.push_back(stof(getTrimmedLine(s)));
    return result;
}

vector<float> TorchUtils::CfgParser::readFloatListOpt(string compact)
{
    auto split = splitCompact(compact);
    return readFloatListOpt(split.first, split.second);
}

vector<double> TorchUtils::CfgParser::readDoubleListOpt(string section, string option)
{
    vector<string> vs = readStringListOpt(section, option);
    vector<double> result;
    for (string s : vs)
        result.push_back(stod(getTrimmedLine(s)));
    return result;
}

vector<double> TorchUtils::CfgParser::readDoubleListOpt(string compact)
{
    auto split = splitCompact(compact);
    return readDoubleListOpt(split.first, split.second);
}

bool TorchUtils::CfgParser::hasOpt(std::string section, std::string option)
{
    trimLine(section);
    trimLine(option);
    if (config_.find(section) == config_.end())
        return false;
    if (config_[section].find(option) == config_[section].end())
        return false;
    return true;
}

bool TorchUtils::CfgParser::hasOpt(std::string compact)
{
    auto split = splitCompact(compact);
    return hasOpt(split.first, split.second);
}

bool TorchUtils::CfgParser::hasSect(std::string section)
{
    if (config_.find(section) == config_.end())
        return false;
    else
        return true;
}

std::vector<std::string> TorchUtils::CfgParser::readListOfOpts(std::string section)
{
    vector<string> opt_list;
    if (!hasSect(section))
        return opt_list;

    for (auto it = config_.at(section).begin(); it != config_.at(section).end(); ++it)
        opt_list.push_back(it->first);

    return opt_list;
}