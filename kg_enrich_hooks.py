# -*- coding: utf-8 -*-
from typing import Dict, Any
from neo4j import Transaction
from nlp_utils import enrich_job_record

UPSERT_CYPHER = """
MERGE (j:Job {jobKey:$jobKey})
SET j.title         = $title,
    j.jobName       = $jobName,
    j.description   = $jobDescription,
    j.descQuality   = $descQuality,
    j.descSource    = $descSource,
    j.eduNormalized = $eduNormalized,
    j.eduConfidence = $eduConfidence,
    j.eduSource     = $eduSource,
    j.salaryMin     = $salaryMin,
    j.salaryMax     = $salaryMax,
    j.salaryMonth   = $salaryMonth,
    j.salaryText    = $salaryText
WITH j, $eduNormalized AS eduNorm, $eduConfidence AS conf, $eduSource AS src
CALL {
  WITH j, eduNorm, conf, src
  WITH j, eduNorm, conf, src WHERE eduNorm <> ''
  MERGE (e:EducationLevel {name: eduNorm})
  MERGE (j)-[r:REQUIRES_EDU]->(e)
  ON CREATE SET r.source=src, r.confidence=conf, r.ts=timestamp()
  ON MATCH  SET r.confidence = CASE WHEN conf>coalesce(r.confidence,0) THEN conf ELSE r.confidence END,
                 r.source = coalesce(r.source, src)
}
RETURN j
"""

def upsert_job_with_edu(tx: Transaction, rec: Dict[str, Any]):
    params = {
        "jobKey": rec.get("jobKey"),
        "title": rec.get("title") or rec.get("jobName"),
        "jobName": rec.get("jobName") or rec.get("title"),
        "jobDescription": rec.get("jobDescription") or rec.get("description"),
        "descQuality": rec.get("descQuality", 0.0),
        "descSource": rec.get("descSource", ""),
        "eduNormalized": rec.get("eduNormalized", ""),
        "eduConfidence": rec.get("eduConfidence", 0.0),
        "eduSource": rec.get("eduSource", ""),
        "salaryMin": rec.get("salaryMin", 0),
        "salaryMax": rec.get("salaryMax", 0),
        "salaryMonth": rec.get("salaryMonth", 12),
        "salaryText": rec.get("salaryText", ""),
    }
    return tx.run(UPSERT_CYPHER, params)
