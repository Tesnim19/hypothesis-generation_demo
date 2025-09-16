#!/usr/bin/env python3
"""
GWAS Metadata Extractor
Extract phenotype, population, and other metadata from GWAS files
"""

import os
import re
import gzip
import pandas as pd
from pathlib import Path
import argparse
import json
from typing import Dict, List, Optional, Union

class GWASMetadataExtractor:
    
    def __init__(self):
        # UK Biobank field codes (common ones)
        self.ukb_field_codes = {
            '21001': 'BMI',
            '50': 'Standing height',
            '21002': 'Weight',
            '4080': 'Systolic blood pressure',
            '4079': 'Diastolic blood pressure',
            '30690': 'Cholesterol',
            '30760': 'HDL cholesterol',
            '30780': 'LDL cholesterol',
            '23104': 'BMI (automated reading)',
            '21022': 'Age at recruitment',
            '20116': 'Smoking status',
            '1558': 'Alcohol intake frequency',
            '6138': 'Education qualifications',
            '20127': 'Neuroticism score',
            '1717': 'Sleep duration',
            '2443': 'Diabetes diagnosed by doctor',
            '6150': 'Vascular/heart problems diagnosed by doctor',
        }
        
        # Known phenotype patterns from filenames/headers
        self.phenotype_patterns = {
            'BMI': ['bmi', 'body_mass_index', 'obesity'],
            'Height': ['height', 'standing_height'],
            'Weight': ['weight', 'body_weight'],
            'Type 2 Diabetes': ['t2d', 'diabetes', 'type2diabetes', 't2dm'],
            'Coronary Artery Disease': ['cad', 'coronary', 'heart_disease'],
            'Blood Pressure': ['bp', 'systolic', 'diastolic', 'hypertension'],
            'Cholesterol': ['cholesterol', 'ldl', 'hdl', 'triglycerides'],
            'Depression': ['depression', 'mdd', 'depressive'],
            'Schizophrenia': ['scz', 'schizophrenia'],
            'Educational Attainment': ['education', 'ea', 'years_of_schooling'],
            'Intelligence': ['iq', 'intelligence', 'cognitive'],
            'Sleep': ['sleep', 'insomnia', 'sleep_duration'],
            'Smoking': ['smoking', 'cigarettes', 'tobacco'],
            'Alcohol': ['alcohol', 'drinks_per_week'],
        }
        
        # Population/ancestry patterns
        self.population_patterns = {
            'EUR': ['european', 'eur', 'white', 'caucasian'],
            'EAS': ['east_asian', 'eas', 'asian', 'chinese', 'japanese', 'korean'],
            'AFR': ['african', 'afr', 'black', 'africa'],
            'AMR': ['admixed_american', 'amr', 'hispanic', 'latino'],
            'SAS': ['south_asian', 'sas', 'indian'],
            'Multi': ['multi', 'mixed', 'trans_ethnic', 'meta']
        }
        
        # Sample size patterns
        self.sample_size_patterns = [
            r'n[_=]?(\d+)', r'N[_=]?(\d+)', r'sample[_\s]*size[_\s]*[=:]?[_\s]*(\d+)',
            r'(\d+)[_\s]*samples?', r'(\d+)[_\s]*individuals?', r'(\d+)[_\s]*subjects?'
        ]

    def extract_from_filename(self, filepath: str) -> Dict[str, str]:
        """Extract metadata from filename"""
        filename = Path(filepath).stem.lower()
        metadata = {}
        
        # Check for UK Biobank field codes first
        ukb_match = re.search(r'(\d{4,5})', filename)
        if ukb_match:
            field_code = ukb_match.group(1)
            if field_code in self.ukb_field_codes:
                metadata['phenotype'] = self.ukb_field_codes[field_code]
                metadata['cohort'] = 'UK Biobank'
                metadata['ukb_field_code'] = field_code
        
        # If no UKB match, try general patterns
        if 'phenotype' not in metadata:
            for phenotype, patterns in self.phenotype_patterns.items():
                if any(pattern in filename for pattern in patterns):
                    metadata['phenotype'] = phenotype
                    break
        
        # Extract population from filename  
        for population, patterns in self.population_patterns.items():
            if any(pattern in filename for pattern in patterns):
                metadata['population'] = population
                break
        
        # Extract sample size from filename
        for pattern in self.sample_size_patterns:
            match = re.search(pattern, filename)
            if match:
                metadata['sample_size'] = int(match.group(1))
                break
        
        # Common GWAS filename patterns
        if 'ukb' in filename or 'ukbb' in filename or 'biobank' in filename:
            metadata['cohort'] = 'UK Biobank'
        elif 'giant' in filename:
            metadata['cohort'] = 'GIANT'
        elif 'magic' in filename:
            metadata['cohort'] = 'MAGIC'
        elif 'diagram' in filename:
            metadata['cohort'] = 'DIAGRAM'
        elif 'pgc' in filename:
            metadata['cohort'] = 'Psychiatric Genomics Consortium'
        
        # Detect if it's munged
        if 'munged' in filename:
            metadata['preprocessing'] = 'MungeSumstats'
        
        # Detect genome build or version
        if 'v3' in filename:
            metadata['version'] = 'v3'
        if 'grch37' in filename or 'hg19' in filename:
            metadata['genome_build'] = 'GRCh37/hg19'
        elif 'grch38' in filename or 'hg38' in filename:
            metadata['genome_build'] = 'GRCh38/hg38'
            
        # Detect sex
        if 'both_sexes' in filename:
            metadata['sex'] = 'Both sexes'
        elif 'male' in filename and 'female' not in filename:
            metadata['sex'] = 'Male only'
        elif 'female' in filename and 'male' not in filename:
            metadata['sex'] = 'Female only'
        
        return metadata

    def extract_from_header(self, filepath: str, max_lines: int = 100) -> Dict[str, str]:
        """Extract metadata from file headers/comments"""
        metadata = {}
        
        try:
            # Handle compressed files
            if filepath.endswith('.gz'):
                opener = gzip.open
                mode = 'rt'
            else:
                opener = open
                mode = 'r'
            
            with opener(filepath, mode) as f:
                lines_read = 0
                for line in f:
                    if lines_read >= max_lines:
                        break
                    
                    line = line.strip().lower()
                    
                    # Skip empty lines
                    if not line:
                        continue
                    
                    # If line doesn't start with comment char and contains tab/comma, 
                    # we've likely hit the data - stop reading
                    if not line.startswith('#') and not line.startswith('//') and ('\t' in line or ',' in line):
                        break
                    
                    # Extract phenotype from comments
                    if any(keyword in line for keyword in ['trait', 'phenotype', 'outcome']):
                        for phenotype, patterns in self.phenotype_patterns.items():
                            if any(pattern in line for pattern in patterns):
                                metadata['phenotype'] = phenotype
                                break
                    
                    # Extract population
                    if any(keyword in line for keyword in ['population', 'ancestry', 'ethnicity']):
                        for population, patterns in self.population_patterns.items():
                            if any(pattern in line for pattern in patterns):
                                metadata['population'] = population
                                break
                    
                    # Extract sample size
                    for pattern in self.sample_size_patterns:
                        match = re.search(pattern, line)
                        if match:
                            metadata['sample_size'] = int(match.group(1))
                            break
                    
                    # Extract other common metadata
                    if 'build' in line or 'genome' in line:
                        if 'hg19' in line or 'grch37' in line:
                            metadata['genome_build'] = 'GRCh37/hg19'
                        elif 'hg38' in line or 'grch38' in line:
                            metadata['genome_build'] = 'GRCh38/hg38'
                    
                    lines_read += 1
        
        except Exception as e:
            print(f"Error reading file headers: {e}")
        
        return metadata

    def extract_from_columns(self, filepath: str, nrows: int = 1000) -> Dict[str, str]:
        """Extract metadata from column names and data"""
        metadata = {}
        
        try:
            # Try to read first few rows to understand structure
            if filepath.endswith('.gz'):
                df = pd.read_csv(filepath, compression='gzip', nrows=nrows, sep=None, engine='python')
            else:
                df = pd.read_csv(filepath, nrows=nrows, sep=None, engine='python')
            
            columns = [col.lower() for col in df.columns]
            metadata['columns'] = list(df.columns)
            metadata['num_variants'] = len(df)
            
            # Check for standard GWAS columns
            standard_columns = {
                'variant_id': ['snp', 'rsid', 'variant', 'variant_id', 'markername'],
                'chromosome': ['chr', 'chromosome', '#chr', 'chrom'],
                'position': ['pos', 'position', 'bp', 'base_pair_location'],
                'effect_allele': ['a1', 'effect_allele', 'allele1', 'ea'],
                'other_allele': ['a2', 'other_allele', 'allele2', 'nea', 'ref_allele'],
                'beta': ['beta', 'effect', 'b', 'log_odds'],
                'se': ['se', 'stderr', 'standard_error'],
                'pvalue': ['p', 'pvalue', 'p_value', 'pval'],
                'sample_size': ['n', 'n_total', 'sample_size', 'neff']
            }
            
            detected_format = {}
            for standard_col, patterns in standard_columns.items():
                for col in columns:
                    if any(pattern == col for pattern in patterns):
                        detected_format[standard_col] = col
                        break
            
            metadata['detected_format'] = detected_format
            metadata['format_type'] = self._determine_format_type(detected_format)
            
            # Try to infer sample size from N column if available
            if 'sample_size' in detected_format:
                n_col = detected_format['sample_size']
                original_col = next(col for col in df.columns if col.lower() == n_col)
                sample_sizes = df[original_col].dropna()
                if len(sample_sizes) > 0:
                    metadata['sample_size'] = int(sample_sizes.iloc[0])
            
        except Exception as e:
            print(f"Error reading file data: {e}")
        
        return metadata

    def _determine_format_type(self, detected_format: Dict[str, str]) -> str:
        """Determine the likely format type of the GWAS file"""
        if 'beta' in detected_format and 'se' in detected_format:
            return 'Linear regression (Beta/SE)'
        elif 'log_odds' in detected_format or ('beta' in detected_format and 'pvalue' in detected_format):
            return 'Logistic regression (OR/Beta)'
        elif 'pvalue' in detected_format:
            return 'Association testing'
        else:
            return 'Unknown format'

    def extract_metadata(self, filepath: str, detailed: bool = True) -> Dict[str, Union[str, int, Dict]]:
        """Extract all available metadata from GWAS file"""
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        print(f"Analyzing GWAS file: {filepath}")
        
        # Initialize metadata
        metadata = {
            'file_path': filepath,
            'file_size_mb': round(os.path.getsize(filepath) / (1024*1024), 2)
        }
        
        # Extract from filename
        filename_metadata = self.extract_from_filename(filepath)
        metadata.update(filename_metadata)
        print(f"From filename: {filename_metadata}")
        
        # Debug: print the actual filename being processed
        print(f"Debug: Processing filename: '{Path(filepath).stem.lower()}'")
        
        # Extract from headers
        header_metadata = self.extract_from_header(filepath)
        metadata.update(header_metadata)
        print(f"From headers: {header_metadata}")
        
        if detailed:
            # Extract from data columns
            column_metadata = self.extract_from_columns(filepath)
            metadata.update(column_metadata)
            print(f"From columns: {dict(column_metadata)}")
        
        return metadata

    def extract_multiple_files(self, file_paths: List[str], output_file: Optional[str] = None) -> List[Dict]:
        """Extract metadata from multiple GWAS files"""
        results = []
        
        for filepath in file_paths:
            try:
                metadata = self.extract_metadata(filepath, detailed=True)
                results.append(metadata)
                print(f"✓ Processed: {Path(filepath).name}")
            except Exception as e:
                print(f"✗ Failed to process {filepath}: {e}")
                results.append({'file_path': filepath, 'error': str(e)})
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to: {output_file}")
        
        return results

    def print_summary(self, metadata: Dict):
        """Print a formatted summary of extracted metadata"""
        print("\n" + "="*60)
        print("GWAS METADATA SUMMARY")
        print("="*60)
        
        # Basic info
        print(f"File: {Path(metadata['file_path']).name}")
        print(f"Size: {metadata.get('file_size_mb', 'Unknown')} MB")
        
        # Key metadata
        key_fields = ['phenotype', 'population', 'cohort', 'sample_size', 'genome_build']
        for field in key_fields:
            if field in metadata:
                print(f"{field.replace('_', ' ').title()}: {metadata[field]}")
        
        # Format info
        if 'format_type' in metadata:
            print(f"Format Type: {metadata['format_type']}")
        
        if 'num_variants' in metadata:
            print(f"Number of Variants: {metadata['num_variants']:,}")
        
        # Column mapping
        if 'detected_format' in metadata and metadata['detected_format']:
            print(f"\nDetected Columns:")
            for standard, actual in metadata['detected_format'].items():
                print(f"  {standard}: {actual}")


def main():
    parser = argparse.ArgumentParser(description="Extract metadata from GWAS files")
    parser.add_argument("files", nargs="+", help="GWAS file(s) to analyze")
    parser.add_argument("--output", "-o", help="Output JSON file for results")
    parser.add_argument("--detailed", action="store_true", 
                       help="Include detailed column analysis (slower)")
    
    args = parser.parse_args()
    
    extractor = GWASMetadataExtractor()
    
    if len(args.files) == 1:
        # Single file
        metadata = extractor.extract_metadata(args.files[0], detailed=args.detailed)
        extractor.print_summary(metadata)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            print(f"\nResults saved to: {args.output}")
    else:
        # Multiple files
        results = extractor.extract_multiple_files(args.files, args.output)
        
        print(f"\n\nSUMMARY: Processed {len(results)} files")
        for result in results:
            if 'error' not in result:
                phenotype = result.get('phenotype', 'Unknown')
                population = result.get('population', 'Unknown') 
                sample_size = result.get('sample_size', 'Unknown')
                print(f"  {Path(result['file_path']).name}: {phenotype} | {population} | N={sample_size}")


if __name__ == "__main__":
    main()