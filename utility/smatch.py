import sys;
from score.core import fscore, intersect;
from smatch.smatch import get_amr_match;

def tuples(graph, prefix):
  #
  # mimicry of get_triples() in amr.py
  #  
  mapping = dict();
  instances = [];
  relations = [];
  attributes = [];
  for node in graph.nodes:
    mapping[node.id] = name = prefix + str(node.id);
    if node.label:
      instances.append(("instance", name, node.label));
    if node.is_top:
      attributes.append(("TOP", name, node.label if node.label else ""));
    if node.properties and node.values:
      for property, value in zip(node.properties, node.values):
        attributes.append((property, name, value));
  for edge in graph.edges:
    if edge.lab == "domain":
      relations.append(("mod", mapping[edge.tgt], mapping[edge.src]));
    elif edge.lab == "domain-of":
      relations.append(("mod", mapping[edge.src], mapping[edge.tgt]));
    elif edge.lab.endswith("-of-of") \
         or edge.lab.endswith("-of") and edge.lab not in {"consist-of" "subset-of" "prep-put-of" "prep-on-behalf-of"} \
                   and not edge.lab.startswith("prep-"):
      normal = edge.lab[:-3]
      relations.append((normal, mapping[edge.tgt], mapping[edge.src]));
    else:
      relations.append((edge.lab, mapping[edge.src], mapping[edge.tgt]));
  return instances, attributes, relations;
        
def evaluate(golds, systems, format = "json", limit = 5, trace = 0):
  if not limit: limit = 5;
  tg = ts = tc = n = 0;
  gprefix = "g"; sprefix = "s";
  scores = dict() if trace else None;
  for gold, system in intersect(golds, systems):
    id = gold.id;
    ginstances, gattributes, grelations = tuples(gold, gprefix);
    sinstances, sattributes, srelations = tuples(system, sprefix);
    correct, gold, system \
      = get_amr_match(None, None, gold.id, limit = limit,
                      instance1 = ginstances, attributes1 = gattributes,
                      relation1 = grelations, prefix1 = gprefix,
                      instance2 = sinstances, attributes2 = sattributes,
                      relation2 = srelations, prefix2 = sprefix);
    tg += gold; ts += system; tc += correct;
    n += 1;
    if trace:
      if id in scores:
        print("smatch.evaluate(): duplicate graph identifier: {}"
              "".format(id), file = sys.stderr);
      scores[id] = {"g": gold, "s": system, "c": correct};
      if trace > 1:
        p, r, f = fscore(gold, system, correct);
        print("G: {}; S: {}; C: {}; P: {}; R: {}; F: {}"
              "".format(gold, system, correct, p, r, f), file = sys.stderr);
        if f != 1.0:
          print("gold instances: {}\ngold attributes {}\ngold relations: {}"
            "".format(ginstances, gattributes, grelations), file=sys.stderr);
          print("system instances: {}\nsystem attributes {}\nsystem relations: {}"
            "".format(sinstances, sattributes, srelations), file=sys.stderr);
    
  p, r, f = fscore(tg, ts, tc);
  result = {"n": n, "g": tg, "s": ts, "c": tc, "p": p, "r": r, "f": f};
  if trace: result["scores"] = scores;
  return result;
