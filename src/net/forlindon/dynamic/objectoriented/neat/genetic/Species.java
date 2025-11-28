package net.forlindon.dynamic.objectoriented.neat.genetic;

import java.util.ArrayList;
import java.util.List;

public class Species implements Comparable<Species> {

    public Genome reference;
    public List<Genome> genomes;

    SpeciesManager speciesManager;

    public Species(SpeciesManager speciesManager, Genome reference) {
        this.speciesManager = speciesManager;
        this.genomes = new ArrayList<>();
        this.reference = reference;
    }

    public boolean contains(Genome genome) {
        return this.reference.calcDelta(genome) < SpeciesManager.SPECIATION_BORDER;
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
                this.genomes.remove(g);
            }
        }
        return out;
    }

    public double averageFitness() {
        return this.genomes.stream().mapToDouble(Genome::getFitness).sum()/this.genomes.size();
    }

    public Genome sample(double d) {
        double delimiter = this.genomes.stream().map(Genome::getFitness).toList().get((int)(this.genomes.size()*(1-d)));
        List<Genome> filtered = this.genomes.stream().filter(genome -> genome.fitness >= delimiter).toList();
        return filtered.get((int)(Math.random()*filtered.size()));
    }

    public Genome getFittest() {
        return this.genomes.stream().sorted(new GenomeFitnessComparator()).toList().getLast();
    }

    public Genome getSecondFittest() {
        return this.genomes.stream().sorted(new GenomeFitnessComparator()).toList().get(this.genomes.size()-2);
    }

    public void removeWeakest() {
        this.genomes.sort(new GenomeFitnessComparator());
        int removeN = (this.genomes.size()-2)/2;
        for (int i = 0; i < removeN; i++) {
            this.genomes.removeFirst();
        }
    }

    public double getFitness() {
        return this.genomes.stream().mapToDouble(Genome::getFitness).sum();
    }

    public double calcMaxDelta() {
        double max = 0;
        for (Genome genome : genomes) {
            max = Math.max(this.reference.calcDelta(genome), max);
        }
        return max;
    }
}
