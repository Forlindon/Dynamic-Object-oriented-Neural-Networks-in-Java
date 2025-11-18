package net.forlindon.dynamic.objectoriented.neat.genetic;

import java.util.*;
import java.util.function.Function;

public class SpeciesManager {

    public static final double SPECIATION_BORDER = 1.5;

    List<Species> species;

    private final int MAX_POPULATION;

    public static double c1 = -0.2; // disjoint Genes
    public static double c2 = -0.2; // excess Genes
    public static double c3 = 0.9; // weights
    public static double c4 = 0.1; // nodes

    public SpeciesManager(int maxPopulation) {
        this.MAX_POPULATION = maxPopulation;
        this.species = new ArrayList<>();
    }

    @Override
    public String toString() {
        return this.species.stream().map(species1 -> species1.genomes.size()).toList().toString();
    }

    public void populate() {
        double[] avFit = this.species.stream().mapToDouble(Species::getFitness).toArray();
        double sum = Arrays.stream(avFit).sum();
        for (int i = 0; i < avFit.length; i++) {
            avFit[i] /= sum;
        }
        while (getPopulationSize() < MAX_POPULATION) {
            double offs = 0;
            double r = Math.random();
            for (int i = 0; i < avFit.length; i++) {
                offs += avFit[i];
                if (Math.random() < offs) {
                    Species species1 = this.species.get(i);
                    if (r < 0.75) species1.add(Genome.crossOver(species1.sample(),species1.sample()));
                    else {
                        Genome g = species1.sample().copy();
                        g.copyValues();
                        species1.add(g);
                    }
                }
            }
        }
    }

    public int getPopulationSize() {
        return this.species.stream().mapToInt(x -> x.genomes.size()).sum();
    }

    public void sort() {
        List<Genome> notPartOfSpecies = new ArrayList<>();
        do {
            for (Species sp : this.species) {
                notPartOfSpecies.addAll(sp.sortOut());
            }
            if (notPartOfSpecies.isEmpty()) return;
            notPartOfSpecies.sort(new GenomeComparator());
            Genome reference = notPartOfSpecies.get(notPartOfSpecies.size() / 2);
            Species newSpecies = new Species(this, reference);
            newSpecies.genomes.addAll(notPartOfSpecies);
            this.species.add(newSpecies);
            notPartOfSpecies.clear();
            notPartOfSpecies.addAll(newSpecies.sortOut());
        }
        while (!notPartOfSpecies.isEmpty());
    }

    public void evaluate(Function<Genome, Double> fitnessSupplier) {
        this.species.forEach(x -> x.genomes.forEach(y -> y.fitness = fitnessSupplier.apply(y)/x.genomes.size()));
    }

    public void removeWeakest() {
        this.species.removeIf(species1 -> species1.genomes.isEmpty());
        this.species.forEach(Species::removeWeakest);
    }

    public static void probability(double[] array) {
        normalize(array);
        softmax(array);
    }

    public static void normalize(double[] array) {
        double min = Arrays.stream(array).min().getAsDouble();
        for (int i = 0; i < array.length; i++) {
            array[i]-=min;
        }
        double sum = Arrays.stream(array).sum();
        if (sum == 0) sum++;
        for (int i = 0; i < array.length; i++) {
            array[i]/=sum;
        }
    }

    public static void softmax(double[] array) {
        for (int i = 0; i < array.length; i++) {
            array[i]=Math.exp(array[i]);
        }
        double sum = Arrays.stream(array).sum();
        for (int i = 0; i < array.length; i++) {
            array[i]/=sum;
        }
    }

    public void init(Species init) {
        this.species.add(init);
        Genome g = init.genomes.getFirst();
        g.fitness = 1e-5;
        while (getPopulationSize() < MAX_POPULATION) {
            Genome x = g.copy();
            x.copyValues();
            this.species.getFirst().add(x);
        }
    }

    public void init() {
        this.init(this.species.getFirst());
    }

    public void mutate() {
        this.species.forEach(sp -> sp.genomes.forEach(MutationFactory::mutateSelf));
    }

    public double averageFitness() {
        return this.species.stream().mapToDouble(Species::averageFitness).sum()/this.species.size();
    }

    public Genome getFittest() {
        List<Genome> genomes = new ArrayList<>(this.species.stream().mapToInt(value -> value.genomes.size()).sum());
        for (Species sp : this.species) {
            genomes.addAll(sp.genomes);
        }
        genomes.sort(new GenomeFitnessComparator());
        return genomes.getLast();
    }

    public int getSpecies() {
        return this.species.size();
    }

    public double getAverageFitness() {
        return this.species.stream().mapToDouble(Species::averageFitness).sum()/this.species.size();
    }
}
