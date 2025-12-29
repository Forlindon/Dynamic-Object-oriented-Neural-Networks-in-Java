package net.forlindon.dynamic.objectoriented.neat.genetic;

import net.forlindon.dynamic.objectoriented.neat.connection.BaseNeatConnection;
import net.forlindon.dynamic.objectoriented.neat.knot.BaseNeatKnot;
import net.forlindon.dynamic.objectoriented.neural.networks.connection.Connection;
import net.forlindon.dynamic.objectoriented.neural.networks.knot.Knot;

import java.util.List;

public final class MutationFactory {

    boolean nodes = false;
    boolean genes = false;
    boolean weights = true;
    boolean activation = false;

    public Genome mutate(Genome g) {
        Genome genome = g.copy();
        mutateSelf(genome);
        return genome;
    }

    private void mutateWeights(Genome genome) {
        for (BaseNeatConnection baseNeatConnection : genome.GENES.values()) {
            if (Math.random() < baseNeatConnection.getMutationRate()) {
                baseNeatConnection.mutate();
            }
        }
    }

    private void mutateBias(Genome genome) {
        for (BaseNeatKnot baseNeatKnot : genome.NODES.values()) {
            if (Math.random() < baseNeatKnot.getMutationRate()) {
                baseNeatKnot.mutate();
            }
        }
    }

    private void mutateActivationFunction(Genome genome) {
        for (BaseNeatKnot baseNeatKnot : genome.NODES.values()) {
            if (baseNeatKnot.id() != genome.LAYER.getFirst() && baseNeatKnot.id() != genome.LAYER.getLast() && BaseNeatKnot.getMutationActivationRate() < Math.random()) {
                BaseNeatKnot.mutateActivation(baseNeatKnot);
            }
        }
    }

    private void mutateNodeAdd(Genome genome) {
        BaseNeatConnection baseNeatConnection = genome.sampleGene();

        // Layer ID
        List<Integer> layers = genome.LAYER;
        int idSrc = baseNeatConnection.src().id();
        int idDest = baseNeatConnection.dest().id();
        int idxA = layers.indexOf(idSrc);
        int idxB = layers.indexOf(idDest);
        int idx;
        if (idxB-idxA == 1) {
            idx = genome.innovationSource.getNext();
            layers.add(idxB, idx);
        }
        else {
            idx = layers.get(idxA+1);
        }

        BaseNeatKnot baseNeatKnot = new BaseNeatKnot(genome.innovationSource, idx);
        baseNeatKnot.OUT = SpeciesManager.getDefaultActivation();
        List<Connection> insert = baseNeatConnection.insert(baseNeatKnot, (a, b) -> new BaseNeatConnection(genome.innovationSource, a, b));
        genome.GENES.put(((BaseNeatConnection)insert.getFirst()).inov(), (BaseNeatConnection) insert.getFirst());
        genome.GENES.put(((BaseNeatConnection)insert.getLast()).inov(), (BaseNeatConnection) insert.getLast());
        genome.NODES.put(baseNeatKnot.inov(), baseNeatKnot);
    }

    private void mutateAddGene(Genome genome) {
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
                genome.GENES.put(baseNeatConnection.inov(),baseNeatConnection);
            }
        }
    }

    public void mutateSelf(Genome genome) {
        // Shift the weights
        if (weights && Math.random() < genome.getGeneMutationRate()) mutateWeights(genome);
        // Shift the bias
        if (weights && Math.random() < genome.getNodeMutationRate()) mutateBias(genome);
        // Change Activation Function
        if (activation && Math.random() < genome.getAcMutationRate() && genome.LAYER.size() > 2) mutateActivationFunction(genome);
        // Change Connection Structure
        if (genes && Math.random() < genome.getGeneMutationAddRate()) mutateAddGene(genome);
        if (genes && Math.random() < genome.getGeneMutationAddRate()) mutateAddGene(genome);
        if (genes && Math.random() < genome.getGeneMutationAddRate()) mutateAddGene(genome);
        // Change Node Structure
        if (nodes && Math.random() < genome.getNodeMutationAddRate()) mutateNodeAdd(genome);
    }

    public static double N() {
        double u1 = Math.random();
        double u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    public void setNodesAdaptation(boolean nodes) {
        this.nodes = nodes;
    }

    public void setGenesAdaptation(boolean genes) {
        this.genes = genes;
    }

    public void setWeightsAdaptation(boolean weights) {
        this.weights = weights;
    }

    public void setActivationAdaptation(boolean activation) {
        this.activation = activation;
    }

    public boolean isAdaptingNodes() {
        return nodes;
    }

    public boolean isAdaptingGenes() {
        return genes;
    }

    public boolean isAdaptingWeights() {
        return weights;
    }

    public boolean isAdaptingActivation() {
        return activation;
    }
}
