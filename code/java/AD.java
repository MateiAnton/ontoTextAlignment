import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.*;
import org.semanticweb.owlapi.reasoner.OWLReasoner;
import uk.ac.manchester.cs.owl.owlapi.OWLDataFactoryImpl;
import uk.ac.manchester.cs.factplusplus.owlapiv3.FaCTPlusPlusReasoner;
import uk.ac.manchester.cs.factplusplus.owlapiv3.FaCTPlusPlusReasonerFactory;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Set;

public class AD {

    private static String atomDirPath = "/path/to/atom_decomposition/";

    public static void main(String[] args) {
        try {
            OWLOntologyManager manager = OWLManager.createOWLOntologyManager();
            File file = new File("ontologyFile");
            Set<OWLOntology> ontologySet = manager.loadOntologyFromOntologyDocument(file).getImportsClosure();
            OWLOntology ontology = manager.createOntology();
            for (OWLOntology ont : ontologySet) {
                manager.addAxioms(ontology, ont.getAxioms());
            }
        } catch (OWLOntologyCreationException | FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    public static void printAtomicDecomposition(OWLOntologyManager manager, OWLOntology o, boolean useSemanticLocality, int moduleType, boolean saveAtoms, boolean printAxioms) {
        OWLReasoner reasoner = new FaCTPlusPlusReasonerFactory().createReasoner(o);
        FaCTPlusPlusReasoner fpp = (FaCTPlusPlusReasoner) reasoner;

        try {
            int nAtoms = fpp.getAtomicDecompositionSize(useSemanticLocality, moduleType);
            PrintWriter printDirectDependenciesOutputFile = new PrintWriter(atomDirPath + "atom-direct-dependencies_v1.txt");
            PrintWriter printBasicStatisticsOutputFile = new PrintWriter(atomDirPath + "atom-basic-statistics_v1.txt");
            PrintWriter printAxiomsOutputFile = new PrintWriter(atomDirPath + "atom-axioms_v1.txt");

            for (int i = 0; i < nAtoms; i++) {
                Set<OWLAxiom> atomAxioms = fpp.getAtomAxioms(i);
                OWLOntology atomOntology = manager.createOntology(atomAxioms);
                System.out.println("Atom " + i + " depends on: " + Arrays.toString(fpp.getAtomDependents(i)));

                if (printAxioms) {
                    System.out.println("Atom " + i + " has axioms: ");
                    printAxiomsOutputFile.println("Atom " + i + " has axioms: ");
                    Iterator<OWLAxiom> aItr = atomAxioms.iterator();
                    while (aItr.hasNext()) {
                        OWLAxiom axiom = aItr.next();
                        printAxiomsOutputFile.println("  - Axiom : " + axiom);
                    }
                    printAxiomsOutputFile.println();
                }
                if (saveAtoms) {
                    saveOntologyToPhysicalIRI(manager, atomOntology, i);
                }
                manager.removeOntology(atomOntology);
            }
            printDirectDependenciesOutputFile.close();
            printBasicStatisticsOutputFile.close();
            printAxiomsOutputFile.close();
        } catch (FileNotFoundException | OWLOntologyCreationException e) {
            e.printStackTrace();
        }
    }

    private static void saveOntologyToPhysicalIRI(OWLOntologyManager manager, OWLOntology ontology, int atomIndex) {
        IRI physicalIRIforRDFXMLSyntax = IRI.create(atomDirPath + "atom-" + atomIndex + ".owl");
        try {
            manager.saveOntology(ontology, new RDFXMLOntologyFormat(), physicalIRIforRDFXMLSyntax);
        } catch (OWLOntologyStorageException e) {
            System.err.println("Error saving ontology: " + e.getLocalizedMessage());
        }
    }
}
