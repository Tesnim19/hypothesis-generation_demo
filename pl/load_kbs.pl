:- style_check(-discontiguous).
:- dynamic(user:file_search_path/2).
% :- dynamic(bgc/1).
% :- multifile(bgc/1).
:- multifile(user:file_search_path/2).
% Node predicates
:- multifile gene/1.
:- multifile protein/1.
:- multifile transcript/1.
:- multifile exon/1.
:- multifile snp/1.
:- multifile structural_variant/1.
:- multifile sequence_variant/1.
:- multifile enhancer/1.
:- multifile promoter/1.
:- multifile super_enhancer/1.
:- multifile non_coding_rna/1.
:- multifile pathway/1.
:- multifile regulatory_region/1.
:- multifile transcription_binding_site/1.
:- multifile go/1.
:- multifile uberon/1.
:- multifile clo/1.
:- multifile cl/1.
:- multifile efo/1.
:- multifile bto/1.
:- multifile motif/1.
:- multifile tad/1.

% Edge predicates
:- multifile transcribed_to/2.
:- multifile transcribed_from/2.
:- multifile translates_to/2.
:- multifile translation_of/2.
:- multifile coexpressed_with/2.
:- multifile interacts_with/2.
:- multifile expressed_in/2.
:- multifile has_part/2.
:- multifile part_of/2.
:- multifile subclass_of/2.
:- multifile capable_of/2.
:- multifile genes_pathways/2.
:- multifile parent_pathway_of/2.
:- multifile child_pathway_of/2.
:- multifile go_gene_product/2.
:- multifile belongs_to/2.
:- multifile associated_with/2.
:- multifile regulates/2.
:- multifile eqtl_association/2.
:- multifile closest_gene/2.
:- multifile upstream_gene/2.
:- multifile downstream_gene/2.
:- multifile in_gene/2.
:- multifile in_ld_with/2.
:- multifile lower_resolution/2.
:- multifile located_on_chain/2.
:- multifile tfbs_snp/2.
:- multifile tf_snp/2.
:- multifile binds_to/2.
:- multifile in_tad_region/2.
:- multifile activity_by_contact/2.
:- multifile chromatin_state/2.
:- multifile in_dnase_I_hotspot/2.
:- multifile histone_modification/2.
:- multifile regulates/2.
:- multifile distance/2.
:- multifile pgboost/2.

% Properties
:- multifile chr/2.
:- multifile start/2.
:- multifile end/2.
:- multifile alt/2.
:- multifile ref/2.
:- multifile gene_name/2.
:- multifile transcript_name/2.
:- multifile transcript_id/2.
:- multifile transcript_type/2.
:- multifile score/2.
:- multifile biological_context/2.
:- multifile caf_ref/2.
:- multifile caf_alt/2.
:- multifile term_name/2.
:- multifile term_name/3.
:- multifile term_name/4.
:- multifile term_name/5.
:- multifile term_name/6.
:- multifile term_name/7.
:- multifile term_name/8.
:- multifile term_name/9.
:- multifile raw_cadd_score/2.
:- multifile phred_score/2.
:- multifile detection_method/2.
:- multifile evidence_type/2.
:- multifile slope/2.
:- multifile maf/2.
:- multifile p_value/2.
:- multifile accession_d/2.

% Additional binary predicates
:- multifile synonyms/2.
:- multifile rel_type/2.


load_with_time(Files, FileName) :-
    format("Loading ~w...~n", [FileName]),
    time(consult(Files)),
    format("Loaded ~w!~n", [FileName]).


% user:file_search_path(prolog_out_v2,'/mnt/prolog_out_v2').
% user:file_search_path(prolog_out_v3,'/mnt/hdd_2/abdu/prolog_out_v3').
user:file_search_path(prolog_out,'/mnt/hdd_1/biocypher-kg/output/prolog_out_v4').
user:file_search_path(prolog_out_v3,'/mnt/hdd_1/biocypher-kg/output/prolog_out_v4').
user:file_search_path(gene, prolog_out('gencode/gene')).
user:file_search_path(exon, prolog_out('gencode/exon')).
user:file_search_path(transcript, prolog_out('gencode/transcript')).
user:file_search_path(uniprot, prolog_out('uniprot')).
user:file_search_path(gene_ontology, prolog_out('gene_ontology')).
user:file_search_path(gaf, prolog_out('gaf')).
% user:file_search_path(cellxgene, prolog_out('cellxgene')).
user:file_search_path(tadmap, prolog_out_v3('tadmap')).
user:file_search_path(tflink, prolog_out_v3('tflink')).
%user:file_search_path(roadmap_dhs, prolog_out('roadmap/dhs')).
%user:file_search_path(roadmap_h3_mark, prolog_out('roadmap/h3_mark')).

user:file_search_path(refseq, prolog_out_v3('refseq')).
user:file_search_path(eqtl, prolog_out_v3('gtex/eqtl')).
user:file_search_path(abc, prolog_out_v3('abc')).
user:file_search_path(cell_line_ontology, prolog_out('cell_line_ontology')).
user:file_search_path(uberon, prolog_out('uberon')).
user:file_search_path(efo, prolog_out('experimental_factor_ontology')).
user:file_search_path(bto, prolog_out('brenda_tissue_ontology')).
user:file_search_path(cadd, prolog_out('cadd')).
user:file_search_path(dbsnp, prolog_out_v3('dbsnp')).
user:file_search_path(dbsuper, prolog_out_v3('dbsuper')).
user:file_search_path(enhancer_atlas, prolog_out_v3('enhancer_atlas')).
user:file_search_path(epd, prolog_out('epd')).
user:file_search_path(fabian, prolog_out('fabian')).
user:file_search_path(peregrine, prolog_out_v3('peregrine')).
user:file_search_path(top_ld_eur, prolog_out('top_ld/EUR')).
user:file_search_path(tfbs, prolog_out_v3('tfbs')).
user:file_search_path(tf_snp, prolog_out('tf_snp')).
user:file_search_path(pgboost, prolog_out_v3('pgboost')).
user:file_search_path(enhancer_ccre, prolog_out('ccre/enhancer_ccre')).
user:file_search_path(promoter_ccre, prolog_out('ccre/promoter_ccre')).


load_atomspace :- 
    %load_with_time([transcript(nodes), transcript(edges)], "gencode transcripts"),
    load_with_time([gene(nodes)], "gencode genes"),
    % load_with_time([exon(nodes)], "gencode exons"),
    %load_with_time([uniprot(nodes), uniprot(edges)], "uniprot"),
    load_with_time([eqtl(edges)], "gtex eqtl"),
    %load_with_time([gene_ontology(nodes), gene_ontology(edges)], "gene ontology"),
    %load_with_time([gaf(edges)], "go gene product"),
    % % load_with_time([cellxgene(edges)], ),
    load_with_time([tadmap(nodes), tadmap(edges)], "tadmap"),
    load_with_time([refseq(edges)], "refseq"),
    load_with_time([abc(edges)], "abc"),
    %load_with_time([cell_line_ontology(nodes), cell_line_ontology(edges)], "cell_line ontology"),
    %load_with_time([uberon(nodes), uberon(edges)], "uberon"),
    load_with_time([efo(nodes), efo(edges)], "experimental factor ontology"),
    %load_with_time([bto(nodes), bto(edges)], "brenda tissue ontology"),
    % load_with_time([cadd(nodes)], "cadd"),
    load_with_time([dbsnp(nodes)], "dbsnp"),
    load_with_time([dbsuper(nodes), dbsuper(edges)], "dbsuper"),
    load_with_time([enhancer_ccre(nodes), enhancer_ccre(edges)], "enhancer ccre"),
    load_with_time([promoter_ccre(nodes), promoter_ccre(edges)], "promoter ccre"),
    load_with_time([enhancer_atlas(nodes), enhancer_atlas(edges)], "enhancer atlas"),
    load_with_time([peregrine(nodes), peregrine(edges)], "peregrine"),
    load_with_time([epd(nodes), epd(edges)], "epd"),
    load_with_time([tflink(edges)], "tflink"),
    load_with_time([tfbs(nodes), tfbs(edges)], "tfbs"),
    load_with_time([tf_snp(edges)], "tf_snp"),
    load_with_time([pgboost(edges)], "pgboost").
    load_with_time([roadmap_chromatin_state(edges)], "roadmap chromatin state"),
    load_with_time([roadmap_dhs(edges)], "roadmap dhs"),
    load_with_time([roadmap_h3_mark(edges)], "roadmap h3 mark"),

