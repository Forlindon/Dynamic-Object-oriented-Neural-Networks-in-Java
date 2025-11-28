package net.forlindon.dynamic.objectoriented.neat.genetic;

import net.forlindon.dynamic.objectoriented.neat.connection.BaseNeatConnection;
import net.forlindon.dynamic.objectoriented.neat.knot.BaseNeatKnot;
import net.forlindon.dynamic.objectoriented.neat.layer.BaseNeatLayer;
import net.forlindon.dynamic.objectoriented.neural.networks.knot.Knot;

import java.util.*;
import java.util.stream.Collectors;

public class Genome {

    Map<Integer, BaseNeatKnot> NODES;
    Map<Integer, BaseNeatConnection> GENES;
    List<Integer> LAYER;
    InnovationSource innovationSource;
    double fitness = 0;

    public Genome(BaseNeatLayer l) {
        this.NODES = new HashMap<>(l.getKNOTS().stream().map(x->(BaseNeatKnot)x).collect(
                Collectors.toMap(
                        BaseNeatKnot::inov,
                        x -> x
                )
        ));
        this.GENES=new HashMap<>(l.getGens().stream().collect(
                Collectors.toMap(
                        BaseNeatConnection::inov,
                        x -> x
                )
        ));
        this.LAYER = l.getKNOTS().stream().map(Knot::id).distinct().collect(Collectors.toList());
        this.innovationSource = l.PARAM_SRC;
    }

    protected Genome(InnovationSource innovationSource, Map<Integer, BaseNeatKnot> nodes, Map<Integer, BaseNeatConnection> genes, List<Integer> layer, double fitness) {
        this.innovationSource = innovationSource;
        this.NODES = new HashMap<>(nodes);
        this.GENES = new HashMap<>(genes);
        this.LAYER = new ArrayList<>(layer);
        this.fitness = fitness;
        copyValues();
    }

    public void setFitness(double d) {
        this.fitness = d;
    }

    public double getFitness() {
        return this.fitness;
    }

    @Override
    public String toString() {
        return "Genome{\n" +
                "NODES=" + NODES.size() +
                ",\nGENES=" + GENES.size() +
                "\n}";
    }

    public String info() {
        return String.format("Fit: %g, Layer: %d, Genes: %d, Nodes: %d", this.fitness, this.LAYER.size(), this.GENES.size(), this.NODES.size());
    }

    public double calcDelta(Genome other) {
        return calcDelta(other, SpeciesManager.c1, SpeciesManager.c2, SpeciesManager.c3);
    }

    public double calcDelta(Genome other, double c1, double c2, double c3) {
        int n = getSize(other);
        int d = getDisjointGens(other);
        int e = getExcessGens(other);
        double w = normalizedGenes(other);
        return (c1 * e + c2 * d)/n + c3 * w;
    }

    public int getSize(Genome other) {
        return Math.max(this.GENES.size(),other.GENES.size());
    }

    public int getDisjointGens(Genome other) {
        return getDisjointAndExcess(this,other,false);
    }

    public int getExcessGens(Genome other) {
        return getDisjointAndExcess(this,other, true);
    }

    public static int getDisjointAndExcess(Genome g1, Genome g2, boolean excess) {
        int maxG1 = Collections.max(g1.GENES.keySet());
        int maxG2 = Collections.max(g2.GENES.keySet());
        int minG1 = Collections.min(g1.GENES.keySet());
        int minG2 = Collections.min(g2.GENES.keySet());

        int count = 0;

        if (excess) {
            for (Integer innov : g1.GENES.keySet()) {
                if (innov > maxG2) count++;
            }
            for (Integer innov : g2.GENES.keySet()) {
                if (innov > maxG1) count++;
            }
        } else {
            int low = Math.max(minG1, minG2);
            int high = Math.min(maxG1, maxG2);

            for (Integer innov : g1.GENES.keySet()) {
                if (innov >= low && innov <= high && !g2.GENES.containsKey(innov)) {
                    count++;
                }
            }
            for (Integer innov : g2.GENES.keySet()) {
                if (innov >= low && innov <= high && !g1.GENES.containsKey(innov)) {
                    count++;
                }
            }
        }

        return count;
    }

    public double normalizedGenes(Genome other) {
        double sum = 0;
        int n = 0;
        for (Map.Entry<Integer,BaseNeatConnection> entry : this.GENES.entrySet()) {
            BaseNeatConnection c = other.GENES.get(entry.getKey());
            if (c != null) {
                sum += Math.abs(entry.getValue().val - c.val);
                n++;
            }
        }
        return n != 0 ? sum / n : 0;
    }

    private double normalizedNodes(Genome other) {
        double sum = 0;
        int n = 0;
        for (Map.Entry<Integer, BaseNeatKnot> entry : this.NODES.entrySet()) {
            Integer key = entry.getKey();
            if (other.NODES.containsKey(key)) {
                BaseNeatKnot a = entry.getValue();
                BaseNeatKnot b = other.NODES.get(key);
                if (!a.OUT.getClass().equals(b.OUT.getClass())) {
                    sum+=1;
                }
                n++;
            }
        }
        return sum/n;
    }

    public Genome copy() {
        return new Genome(this.innovationSource,this.NODES,this.GENES,this.LAYER, this.fitness);
    }

    public void copyValues() {
        this.GENES.clear();
        for (Map.Entry<Integer, BaseNeatKnot> entry : this.NODES.entrySet()) {
            BaseNeatKnot baseNeatKnot = entry.getValue().copy();
            entry.setValue(baseNeatKnot);
            for (BaseNeatConnection baseNeatConnection : baseNeatKnot.getConnections().stream().map(connection -> (BaseNeatConnection) connection).toList()) {
                this.GENES.put(baseNeatConnection.inov(), baseNeatConnection);
            }
        }
        for (Map.Entry<Integer,BaseNeatConnection> entry : this.GENES.entrySet()) {
            BaseNeatKnot dest = (BaseNeatKnot) entry.getValue().dest();
            entry.getValue().setDest(this.NODES.get(dest.inov()));
        }
    }

    public static Genome crossOver(Genome a, Genome b) {
        Genome leading = (a.fitness > b.fitness ? a : b);
        Genome trailing = leading == b ? a : b;
        leading = leading.copy();
        leading.fitness = (a.fitness+b.fitness)/2;

        Map<Integer,BaseNeatConnection> leadingGenes = leading.GENES;
        Map<Integer,BaseNeatConnection> trailingGenes = trailing.GENES;

        for (Map.Entry<Integer,BaseNeatConnection> entry : leadingGenes.entrySet()) {
            Integer key = entry.getKey();
            if (trailingGenes.containsKey(key)) {
                BaseNeatConnection value = entry.getValue();
                value.val = (value.val+trailingGenes.get(key).val)/2.0;
            }
        }

        Map<Integer,BaseNeatKnot> leadingNodes = leading.NODES;
        Map<Integer,BaseNeatKnot> trailingNodes = trailing.NODES;

        for (Map.Entry<Integer,BaseNeatKnot> entry : leadingNodes.entrySet()) {
            Integer key = entry.getKey();
            if (trailingNodes.containsKey(key)) {
                leadingNodes.get(key).BIAS.val = (leadingNodes.get(key).BIAS.val + trailingNodes.get(key).BIAS.val)/2.0;
            }
        }

        return leading;
    }

    public double getNodeMutationAddRate() {
        return 0.1;
    }
    public double getGeneMutationRate() {
        return 0.8;
    }
    public double getNodeMutationRate() {
        return 0.2;
    }
    public double getGeneMutationAddRate() {
        return 0.5;
    }
    public double getAcMutationRate() {
        return 0.1;
    }

    public double getNodes() {
        return this.NODES.size();
    }

    public BaseNeatKnot sampleNode() {
        List<BaseNeatKnot> list = this.NODES.values().stream().toList();
        return list.get((int)(Math.random()*list.size()));
    }

    public BaseNeatKnot sampleNonOutNode() {
        List<BaseNeatKnot> list = this.NODES.values().stream()
                .filter(baseNeatKnot -> !baseNeatKnot.getConnections().isEmpty())
                .toList();
        return list.get((int)(Math.random()*list.size()));
    }

    public BaseNeatConnection sampleGene() {
        List<BaseNeatConnection> list = this.GENES.values().stream().toList();
        return list.get((int)(Math.random()*list.size()));
    }

    public Map<Integer, BaseNeatKnot> getNODES() {
        return NODES;
    }

    public Map<Integer, BaseNeatConnection> getGENES() {
        return GENES;
    }

    public List<Integer> getLAYER() {
        return LAYER;
    }

    public InnovationSource getInnovationSource() {
        return innovationSource;
    }
}
