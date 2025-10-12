package net.forlindon.dynamic.objectoriented.neat.genetic;

import net.forlindon.dynamic.objectoriented.neat.connection.BaseNeatConnection;
import net.forlindon.dynamic.objectoriented.neat.knot.BaseNeatKnot;
import net.forlindon.dynamic.objectoriented.neat.layer.BaseNeatLayer;

import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Collectors;

public class Genome {

    Map<Integer, BaseNeatKnot> NODES;
    Map<Integer, BaseNeatConnection> GENES;
    InnovationSource innovationSource;

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
        this.innovationSource = l.PARAM_SRC;
    }

    protected Genome(InnovationSource innovationSource, Map<Integer, BaseNeatKnot> nodes, Map<Integer, BaseNeatConnection> genes) {
        this.innovationSource = innovationSource;
        this.NODES = new TreeMap<>(nodes);
        this.GENES = new TreeMap<>(genes);
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
        double w = normalizedGens(other);
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

    public double normalizedGens(Genome other) {
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
        return new Genome(this.innovationSource,this.NODES,this.GENES);
    }
}
