#include "arbitrium/contracts.h"
#include <iostream>
int main(int argc,char** argv) {
  try {
    if(argc!=3)return 2;
    auto task=arbitrium::TaskSpec::from_json(arbitrium::parse_json(arbitrium::read_file(argv[1],4*1024*1024),4*1024*1024));
    std::string line;
    while(std::getline(std::cin,line)) {
      try {
        auto j=arbitrium::parse_json(line,128*1024);
        if(std::string(argv[2])=="request")arbitrium::DecisionRequest::from_json(j,task);
        else if(std::string(argv[2])=="sample")arbitrium::Sample::from_json(j,task);
        else return 2;
        std::cout<<"{\"valid\":true}\n";
      } catch(const std::exception&) {std::cout<<"{\"valid\":false}\n";}
    }
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 3;}
}
