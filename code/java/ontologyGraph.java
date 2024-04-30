import java.io.File;
import java.io.FileWriter;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import org.semanticweb.owlapi.apibinding.OWLManager;
import org.semanticweb.owlapi.model.AxiomType;
import org.semanticweb.owlapi.model.OWLClass;
import org.semanticweb.owlapi.model.OWLClassExpression;
import org.semanticweb.owlapi.model.OWLSubClassOfAxiom;
import org.semanticweb.owlapi.model.OWLOntology;
import org.semanticweb.owlapi.model.OWLOntologyManager;
import org.json.JSONArray;
import org.json.JSONObject;

public class OntologyGraph {

    public static void main(String[] args) {
        String ontologyFile = "/path/to/your/ontology/file.owl";

        try {
            OWLOntologyManager manager = OWLManager.createOWLOntologyManager();
            File file = new File(ontologyFile);
            Set<OWLOntology> ontologySet = manager.loadOntologyFromOntologyDocument(file).getImportsClosure();
            OWLOntology ontology = manager.createOntology();

            // Merging all ontologies
            for (OWLOntology ont : ontologySet) {
                manager.addAxioms(ontology, ont.getAxioms());
            }

            Map<OWLSubClassOfAxiom, Set<OWLClass>> axiomLHSMap = new HashMap<>();
            Map<OWLSubClassOfAxiom, Set<OWLClass>> axiomRHSMap = new HashMap<>();
            Map<OWLSubClassOfAxiom, Set<OWLSubClassOfAxiom>> graph = new HashMap<>();

            Set<OWLSubClassOfAxiom> subclassAxioms = ontology.getAxioms(AxiomType.SUBCLASS_OF);

            // Populating LHS and RHS maps
            for (OWLSubClassOfAxiom axiom : subclassAxioms) {
                OWLClassExpression lhs = axiom.getSubClass();
                OWLClassExpression rhs = axiom.getSuperClass();
                Set<OWLClass> lhsClasses = lhs.getClassesInSignature();
                Set<OWLClass> rhsClasses = rhs.getClassesInSignature();
                axiomLHSMap.put(axiom, lhsClasses);
                axiomRHSMap.put(axiom, rhsClasses);
                graph.put(axiom, new HashSet<>());
            }

            // Creating graph edges
            for (OWLSubClassOfAxiom sourceAxiom : subclassAxioms) {
                Set<OWLClass> sourceRHS = axiomRHSMap.get(sourceAxiom);
                for (OWLSubClassOfAxiom targetAxiom : subclassAxioms) {
                    if (sourceAxiom.equals(targetAxiom)) continue;
                    Set<OWLClass> targetLHS = axiomLHSMap.get(targetAxiom);
                    for (OWLClass cls : sourceRHS) {
                        if (targetLHS.contains(cls)) {
                            graph.get(sourceAxiom).add(targetAxiom);
                            break;
                        }
                    }
                }
            }

            // Convert graph to JSON and write to file
            JSONObject jsonGraph = new JSONObject();
            for (Map.Entry<OWLSubClassOfAxiom, Set<OWLSubClassOfAxiom>> entry : graph.entrySet()) {
                JSONArray targets = new JSONArray();
                for (OWLSubClassOfAxiom target : entry.getValue()) {
                    targets.put(target.toString());  // Adjust as needed
                }
                jsonGraph.put(entry.getKey().toString(), targets);
            }

            try (FileWriter writer = new FileWriter(ontologyFile + "_onto_graph.json")) {
                writer.write(jsonGraph.toString(4));  // Indentation for readability
            }

        } catch (Exception e) {
            System.out.println("Error processing the ontology: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
