package net.forlindon.dynamic.objectoriented.neat.genetic;

import net.forlindon.dynamic.objectoriented.neat.connection.BaseNeatConnection;
import net.forlindon.dynamic.objectoriented.neat.knot.BaseNeatKnot;
import net.forlindon.dynamic.objectoriented.neural.networks.knot.Knot;

import java.util.List;

public final class MutationFactory {

    public Genome mutate(Genome g) {
        Genome genome = g.copy();
        genome.copyValues();
        mutateSelf(genome);
        return genome;
    }

    public static void mutateSelf(Genome genome) {
        // Shift the weights
        if (Math.random() < genome.getGeneMutationRate()) {
            for (BaseNeatConnection baseNeatConnection : genome.GENES.values()) {
                if (Math.random() < baseNeatConnection.getMutationRate()) {
                    baseNeatConnection.mutate();
                }
            }
        }
        // Shift the bias and the activation function
        if (Math.random() < genome.getNodeMutationRate()) {
            for (BaseNeatKnot baseNeatKnot : genome.NODES.values()) {
                if (Math.random() < baseNeatKnot.getMutationRate()) {
                    baseNeatKnot.mutate();
                }
            }
        }
        // Change Activation Function
        if (Math.random() < genome.getAcMutationRate() && genome.LAYER.size() > 2) {
            List<BaseNeatKnot> baseNeatKnots = genome.NODES.values().stream().toList();
            BaseNeatKnot baseNeatKnot;
            do {
                int idx = (int)(Math.random()*baseNeatKnots.size());
                baseNeatKnot = baseNeatKnots.get(idx);
            }
            while (baseNeatKnot == null || genome.LAYER.getLast().equals(baseNeatKnot.id()) || genome.LAYER.getFirst().equals(baseNeatKnot.id()));
            BaseNeatKnot.mutateActivation(baseNeatKnot);
        }
        //Change the structure
        if (Math.random() < genome.getGeneMutationAddRate()) {
            List<Integer> layers = genome.LAYER;
            List<BaseNeatKnot> knots = genome.NODES.values().stream().toList();

            int a = (int) (knots.size() * Math.random());
            int b = (int) (knots.size() * Math.random());

            Knot kA = knots.get(a);
            Knot kB = knots.get(b);
            if (kA.id() != kB.id()) {
                if (layers.indexOf(kA.id()) > layers.indexOf(kB.id())) {
                    kA = kB;
                    kB = knots.get(a);
                }
                if (!kA.isConnectedTo(kB)) {
                    BaseNeatConnection baseNeatConnection = new BaseNeatConnection(genome.innovationSource,kA,kB);
                    kA.add(baseNeatConnection);
                    genome.GENES.put(baseNeatConnection.getInnovationNumber(),baseNeatConnection);
                }
            }
        }
        if (Math.random() < genome.getNodeMutationAddRate()) {
            int r = (int) (genome.LAYER.size() * Math.random());
            if (Math.random() > 0.3 && (r == 0 || r == genome.LAYER.size()-1)) return;
            if (r == 0) genome.LAYER.add(++r, genome.innovationSource.getNext());
            else if (r == genome.LAYER.size() - 1) genome.LAYER.add(r,genome.innovationSource.getNext());
            BaseNeatKnot bk = new BaseNeatKnot(genome.innovationSource, genome.LAYER.get(r));
            genome.NODES.put(bk.getInnovationNumber(), bk);
        }
    }

    public static double N() {
        double u1 = Math.random();
        double u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

}
