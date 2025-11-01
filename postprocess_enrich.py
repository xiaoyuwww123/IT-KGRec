# -*- coding: utf-8 -*-
from neo4j import GraphDatabase
from config import BOLT_URI, DATABASE, USER, PASSWORD, NO_AUTH
from nlp_utils import enrich_job_record
from kg_enrich_hooks import upsert_job_with_edu

FETCH_JOBS = """
MATCH (j:Job)
RETURN j.jobKey AS jobKey,
       j.title  AS title,
       j.jobName AS jobName,
       j.description AS description,
       j.jobDescription AS jobDescription,
       j.detail AS detail,
       j.jd AS jd,
       j.text AS text,
       j.desc AS desc,
       j.content AS content,
       j.requirement AS requirement,
       j.requirements AS requirements,
       j.salaryMin AS salaryMin,
       j.salaryMax AS salaryMax,
       j.salaryMonth AS salaryMonth,
       j.salaryText AS salaryText,
       j.edu AS edu,
       j.education AS education
"""

def main():
    driver = GraphDatabase.driver(BOLT_URI, auth=None if NO_AUTH else (USER, PASSWORD))
    with driver.session(database=DATABASE) as sess:
        rows = sess.run(FETCH_JOBS).data()
        n = 0
        for r in rows:
            rec = {k: r.get(k) for k in r.keys()}
            rec = enrich_job_record(rec)
            sess.execute_write(upsert_job_with_edu, rec)
            n += 1
            if n % 100 == 0:
                print(f"[enrich] processed {n}")
    print("done.")

if __name__ == "__main__":
    main()
