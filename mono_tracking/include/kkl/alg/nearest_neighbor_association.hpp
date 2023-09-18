/**
 * NearestNeighborAssociation.hpp
 * @author koide
 * 15/05/29
**/
#ifndef KKL_NEAREST_NEIGHBOR_ASSOCIATION_HPP
#define KKL_NEAREST_NEIGHBOR_ASSOCIATION_HPP

#include <algorithm>
#include <kkl/alg/data_association.hpp>

using namespace std;

namespace kkl {
	namespace alg {

/******************************************
 * NearestNeighborAssociation
 *
******************************************/
template<typename Tracker, typename Observation, typename Distance>
class NearestNeighborAssociation : public DataAssociation<Tracker, Observation> {
  typedef typename DataAssociation<Tracker, Observation>::Association Association;
public:
  // constructor, destructor
  NearestNeighborAssociation(const Distance& distance)
      : distance(distance)
  {}
  virtual ~NearestNeighborAssociation() {}

  // associate
  // TODO: Use parallel style distance calculating (maybe use cuda)
  std::vector<Association> associate(const std::vector<Tracker>& trackers, const std::vector<Observation>& observations) override {
    if(trackers.empty() || observations.empty()) {
      return std::vector<Association>();
    }

    std::vector<Association> all_associations;
    all_associations.reserve(trackers.size() * 2);  // what if number of observations is larger than trackers???

    for(int i=0; i<trackers.size(); i++) {
      for(int j=0; j<observations.size(); j++) {
        // cout << "Association A" << endl;
        auto dist = distance(trackers[i], observations[j]);
        if(dist) {
          // cout << "Association B" << endl;
          all_associations.push_back(Association(i, j, dist.get()));
        }
      }
    }
    // cout << "Association C" << endl;
    std::sort(all_associations.begin(), all_associations.end());

    std::vector<Association> associations;
    associations.reserve(trackers.size());
    while(!all_associations.empty()) {
      associations.push_back(all_associations.front());
      // remove the association that has the associated tracker/observation
      // why use vector, not list???
      auto remove_loc = std::remove_if(all_associations.begin(), all_associations.end(), [&](const Association& assoc) {
        return assoc.tracker == associations.back().tracker || assoc.observation == associations.back().observation;
      });
      all_associations.erase(remove_loc, all_associations.end());
    }

    return associations;
  }

private:
  Distance distance;
};

	}
}

#endif
