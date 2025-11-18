package net.forlindon.dynamic.objectoriented.neat.genetic;

import net.forlindon.dynamic.objectoriented.neat.connection.BaseNeatConnection;
import net.forlindon.dynamic.objectoriented.neat.knot.BaseNeatKnot;
import net.forlindon.dynamic.objectoriented.neat.layer.BaseNeatLayer;
import net.forlindon.dynamic.objectoriented.neat.layer.NeatLinearLayer;
import net.forlindon.dynamic.objectoriented.neat.layer.SequentialNeatLayer;
import net.forlindon.dynamic.objectoriented.neural.networks.connection.Connection;
import net.forlindon.dynamic.objectoriented.neural.networks.knot.Knot;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Collectors;

public class Genome {

    Map<Integer, BaseNeatKnot> NODES;
    Map<Integer, BaseNeatConnection> GENES;
    List<Integer> LAYER;
    InnovationSource innovationSource;
    double fitness = 0;

    public Genome(BaseNeatLayer l) {
        this.NODES = l.getKNOTS().stream().map(x->(BaseNeatKnot)x).collect(
                Collectors.toMap(
                        BaseNeatKnot::getInnovationNumber,
                        x -> x
                )
        );
        this.GENES=l.getGens().stream().collect(
                Collectors.toMap(
                        BaseNeatConnection::getInnovationNumber,
                        x -> x
                )
        );
        this.LAYER = l.getKNOTS().stream().map(Knot::id).distinct().collect(Collectors.toList());
        this.innovationSource = l.PARAM_SRC;
    }

    protected Genome(InnovationSource innovationSource, Map<Integer, BaseNeatKnot> nodes, Map<Integer, BaseNeatConnection> genes, List<Integer> layer, double fitness) {
        this.innovationSource = innovationSource;
        this.NODES = new TreeMap<>(nodes);
        this.GENES = new TreeMap<>(genes);
        this.LAYER = new ArrayList<>(layer);
        this.fitness = fitness;
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
        return calcDelta(other, SpeciesManager.c1,SpeciesManager.c2,SpeciesManager.c3,SpeciesManager.c4);
    }

    public double calcDelta(Genome other, double c1, double c2, double c3, double c4) {
        int n = getSize(other);
        int e = getDisjointGens(other);
        int c = getExcessGens(other);
        double w = normalizedGenes(other);
        double a = normalizedNodes(other);
        return Math.abs(c1 * e/n + c2 * c/n + c3 * w + c4 * a);
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

    public int getSize(Genome other) {
        return Math.max(this.GENES.size(),other.GENES.size());
    }

    public int getDisjointGens(Genome other) {
        return getNotMatchingGens(this,other);
    }

    public int getExcessGens(Genome other) {
        return getNotMatchingGens(other,this);
    }

    public static int getNotMatchingGens(Genome a, Genome b) {
        int n = 0;
        for (Integer i : a.GENES.keySet()) {
            if (!b.GENES.containsKey(i)) {
                n++;
            }
        }
        return n;
    }

    public double normalizedGenes(Genome other) {
        double sum = 0;
        int n = 0;
        for (Map.Entry<Integer,BaseNeatConnection> entry : this.GENES.entrySet()) {
            BaseNeatConnection c = other.GENES.get(entry.getKey());
            if (c != null) {
                sum += entry.getValue().val - c.val;
                n++;
            }
        }
        return sum / n;
    }

    public Genome copy() {
        return new Genome(this.innovationSource,this.NODES,this.GENES,this.LAYER, this.fitness);
    }

    public void copyValues() {
        for (Map.Entry<Integer, BaseNeatKnot> entry : this.NODES.entrySet()) {
            entry.setValue(entry.getValue().copy());
            for (Connection connection : entry.getValue().getConnections()) {
                this.GENES.put(((BaseNeatConnection) connection).getInnovationNumber(), (BaseNeatConnection) connection);
            }
        }
        for (Map.Entry<Integer,BaseNeatConnection> entry : this.GENES.entrySet()) {
            BaseNeatKnot dest = (BaseNeatKnot) entry.getValue().dest();
            entry.getValue().setDest(this.NODES.get(dest.getInnovationNumber()));
        }
    }

    public static Genome crossOver(Genome a, Genome b) {
        Genome leading = (a.fitness > b.fitness ? a : b).copy();
        leading.copyValues();
        Genome trailing = a.fitness < b.fitness ? a : b;

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
        return 0.02;
    }

    public double getGeneMutationRate() {
        return 0.9;
    }

    public double getNodeMutationRate() {
        return 0.8;
    }

    public double getGeneMutationAddRate() {
        return 0.05;
    }

    public double getAcMutationRate() {
        return 0.01;
    }

    public double getNodes() {
        return this.NODES.size();
    }
}
