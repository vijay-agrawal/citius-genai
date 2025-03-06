from neo4j import GraphDatabase

# 1. Connect to your Neo4j instance.
neo4j_uri = ""
neo4j_user = ""
neo4j_password = ""
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

def run_cypher_query(query: str):
    """Execute a Cypher query and return the results."""
    with driver.session() as session:
        result = session.run(query)
        return [record.data() for record in result]

def clear_graph():
    """Delete all nodes and relationships to avoid duplication errors."""
    clear_query = "MATCH (n) DETACH DELETE n"
    run_cypher_query(clear_query)

def create_sample_data():
    """Create sample nodes and relationships for a healthcare knowledge graph."""
    create_query = """
    CREATE (p:Patient {name: "John Doe", age: 45, gender: "Male"})
    CREATE (d:Doctor {name: "Dr. Smith", specialty: "Cardiology"})
    CREATE (c:Condition {name: "Hypertension", severity: "High"})
    CREATE (m:Medication {name: "Lisinopril", dosage: "10mg", frequency:"Once daily"})
    
    CREATE (p)-[:DIAGNOSED_WITH {diagnosis_date: "2022-05-01"}]->(c)
    CREATE (d)-[:TREATS {since: "2022-06-01"}]->(c)
    CREATE (m)-[:TREATMENT_FOR]->(c)
    """
    run_cypher_query(create_query)

# 3. Main execution: Clear graph, create sample data, then generate and execute the query.
if __name__ == "__main__":
    try:
        clear_graph()
        create_sample_data()
        
        cypher_query = "MATCH (p:Patient {name: 'John Doe'}); RETURN p"
        print("Cypher Query:")
        print(cypher_query, "\n")
        
        results = run_cypher_query(cypher_query)
        print("Query Results:")
        print(results)
    except Exception as e:
        print("An error occurred:", e)