package net.forlindon.dynamic.objectoriented.neat.genetic;

import net.forlindon.dynamic.objectoriented.neat.connection.BaseNeatConnection;
import net.forlindon.dynamic.objectoriented.neat.knot.BaseNeatKnot;
import net.forlindon.dynamic.objectoriented.neat.layer.BaseNeatLayer;
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

    protected Genome(InnovationSource innovationSource, Map<Integer, BaseNeatKnot> nodes, Map<Integer, BaseNeatConnection> genes, List<Integer> layer) {
        this.innovationSource = innovationSource;
        this.NODES = new TreeMap<>(nodes);
        this.GENES = new TreeMap<>(genes);
        this.LAYER = new ArrayList<>(layer);
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
                "NODES=" + NODES +
                ", GENES=" + GENES +
                "\n}";
    }

    public double calcDelta(Genome other) {
        int n = getSize(other);
        int e = getDisjointGens(other);
        int c = getExcessGens(other);
        double w = normalizedGenes(other);
        return (double) e/n + (double) c/n + w;
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
        return new Genome(this.innovationSource,this.NODES,this.GENES,this.LAYER);
    }

    public void copyValues() {
        for (Map.Entry<Integer,BaseNeatKnot> entry : this.NODES.entrySet()) {
            entry.setValue(entry.getValue().copy());
        }
        for (Map.Entry<Integer,BaseNeatConnection> entry : this.GENES.entrySet()) {
            entry.setValue(entry.getValue().copy());
            entry.getValue().setSrc(this.NODES.get(entry.getValue().src().getInnovationNumber()));
            entry.getValue().setDest(this.NODES.get(entry.getValue().dest().getInnovationNumber()));
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
        return leading;
    }

    public double getMutationRate() {
        return 0.001;
    }
}
