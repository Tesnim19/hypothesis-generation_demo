import xml.etree.ElementTree as ET
import html
from typing import List, Dict


class LiteratureParser:
    @staticmethod
    def parse_pubmed_xml(response_text: str) -> List[Dict]:
        """
        Parse PubMed XML response and return a list of normalized articles.
        Each article has: title, abstract, sections, source, id, year, doi.
        """
        articles = []

        try:
            root = ET.fromstring(response_text)

            for article in root.findall(".//PubmedArticle"):
                # PMID 
                pmid = article.findtext(".//PMID")

                # Title 
                title = article.findtext(".//ArticleTitle") or "" 

                # Abstract sections
                abstract = ""
                relevant_sections = []
                irrelevant_labels = {
                    "METHODS", "MATERIALS AND METHODS", "PATIENTS AND METHODS",
                    "METHODOLOGY","FUNDING","ACKNOWLEDGEMENTS","FUNDING INFORMATION",
                    "FINANCIAL SUPPORT","TRIAL REGISTRATION","ETHICS",
                    "ETHICAL CONSIDERATIONS","DATA AVAILABILITY","DATA SHARING",
                    "SUPPLEMENTARY MATERIAL","AUTHOR CONTRIBUTIONS","CONFLICT OF INTEREST",
                    "DISCLOSURES","AUTHOR AFFILIATIONS","KEYWORDS","MESH TERMS",
                    "LIMITATIONS","ABBREVIATIONS"
                }

                abstract_texts = article.findall(".//Abstract/AbstractText")
                for abs_text in abstract_texts:
                    label = abs_text.attrib.get("Label", "ABSTRACT")
                    text = html.unescape("".join(abs_text.itertext())).strip()
                    if text:
                        abstract += (text + " ")
                        if label not in irrelevant_labels:
                            relevant_sections.append({"label": label, "text": text})

                # DOI
                doi = None
                for id_node in article.findall(".//ArticleIdList/ArticleId"):
                    if id_node.attrib.get("IdType") == "doi":
                        doi = id_node.text
                        break

                # Year (if available)
                year = None
                pub_date = article.find(".//PubDate")
                if pub_date is not None:
                    year_text = pub_date.findtext("Year")
                    if year_text and year_text.isdigit():
                        year = int(year_text)

                articles.append(
                    {
                        "title": title,
                        "abstract": abstract,
                        "relevant_sections": relevant_sections,
                        "source": "PubMed",
                        "id": pmid,
                        "year": year,
                        "doi": doi,
                    }
                )

        except Exception as e:
            print(f"[PubMed XML Parser] Error: {e}")

        return articles