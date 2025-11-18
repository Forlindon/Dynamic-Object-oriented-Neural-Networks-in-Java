package net.forlindon.dynamic.objectoriented.neat.genetic;

import java.util.ArrayList;
import java.util.List;

public class Species implements Comparable<Species> {

    Genome reference;
    List<Genome> genomes;

    SpeciesManager speciesManager;

    public Species(SpeciesManager speciesManager, Genome reference) {
        this.speciesManager = speciesManager;
        this.genomes = new ArrayList<>();
        this.genomes.add(reference);
        this.reference = reference;
    }

    public boolean contains(Genome entity) {
        return this.reference.calcDelta(entity) < SpeciesManager.SPECIATION_BORDER;
    }

    public void add(Genome ent) {
        if (contains(ent)) this.genomes.add(ent);
    }

    @Override
    public int compareTo(Species o) {
        return Double.compare(this.reference.fitness,o.reference.fitness);
    }

    public List<Genome> sortOut() {
        List<Genome> out = new ArrayList<>();
        for (int i = 0; i < this.genomes.size(); i++) {
            Genome g = this.genomes.get(i);
            if (!contains(g)) {
                out.add(g);
            }
            this.genomes.remove(g);
        }
        return out;
    }

    public double averageFitness() {
        return this.genomes.stream().mapToDouble(Genome::getFitness).sum()/this.genomes.size();
    }

    public Genome sample() {
        return this.genomes.get((int)(this.genomes.size()*Math.random()));
    }

    public void removeWeakest() {
        double[] array = this.genomes.stream().mapToDouble(Genome::getFitness).sorted().toArray();
        double delimiter = array[(int)(array.length*0.75)];
        this.genomes.removeIf(genome -> genome.fitness < delimiter);
    }

    public void populate(int n) {

    }

    public double getFitness() {
        return this.genomes.stream().mapToDouble(Genome::getFitness).sum();
    }
}
