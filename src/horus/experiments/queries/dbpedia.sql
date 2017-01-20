PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbr: <http://dbpedia.org/resource/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

---------------------------------------------------------------------
--PERSON
---------------------------------------------------------------------
SELECT ?person, ?abstract
FROM <http://dbpedia.org>
WHERE {
    ?person rdf:type dbo:Person .
    ?person dbo:abstract ?abstract .
    FILTER (lang(?abstract) = 'en')
}

---------------------------------------------------------------------
--LOC
---------------------------------------------------------------------
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbr: <http://dbpedia.org/resource/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?location, ?abstract
FROM <http://dbpedia.org>
WHERE {
    ?location rdf:type dbo:Location .
    ?location dbo:abstract ?abstract .
    FILTER (lang(?abstract) = 'en')
}

{?location rdf:type dbo:Location }
    UNION { ?location rdf:type dbo:City} .

dbo:Towns
dbo:Municipality,
dbo:Hill_station,
dbo:Village,
dbo:Suburb,
dbo:Neighborhood,
dbo:NaturalPlace,
dbo:Urban_areas  .
---------------------------------------------------------------------
--ORG
---------------------------------------------------------------------
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbr: <http://dbpedia.org/resource/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?org, ?abstract
FROM <http://dbpedia.org>
WHERE { ?org rdf:type dbo:Organisation  .
      ?org dbo:abstract ?abstract .
    FILTER (lang(?abstract) = 'en')
}
