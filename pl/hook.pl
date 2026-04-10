:- style_check(-discontiguous).
:- dynamic(user:file_search_path/2).
:- multifile(user:file_search_path/2).

:- dynamic seen_predicate/3.

ensure_dynamic_arity(Space,Arity) :- ( current_predicate(Space/Arity)
                                       -> true ; dynamic(Space/Arity) ).



%Define term_expansion/2 to process terms as they are read
%term_expansion(Head, [:- multifile(Declaration), :- discontiguous(Declaration), Term]) :-
%    Head \= begin_of_file,
%    Head \= end_of_file,
%    Head \= :-(_),
%    prolog_load_context(source, Source),
%    Head =.. [Name | Args]
%    length(Args, N),
%    (seen_predicate(Source, Name, Arity) ->
%        false
%    ;   assertz(seen_predicate(Source, Name, Arity)), 
%        Declaration = [Name/Arity]
%            ).
   

term_expansion(Head, Out) :-
    Head \= begin_of_file,
    Head \= end_of_file,
    Head \= :-(_),
    prolog_load_context(source, Source),
    Head =.. [Name | Args],
    length(Args, N),
    (seen_predicate(Source, Name, Arity) ->
      Decls = []
    ; assertz(seen_predicate(Source, Name, Arity)),
      Decls = [:- multifile([Name/N]), :- discontiguous([Name/N])]),
    % append(Decls, [Head], Out).
    atom_string(Space, "&self"),
    Arity is N + 1,
    (seen_predicate(Source, Space, Arity) ->
      Decls2 = []
    ; append(Decls, [:- multifile(Space/Arity), :- dynamic(Space/Arity), :- discontiguous([Space/Arity])], Decls2),
      assertz(seen_predicate(Source, Space, Arity))
    ),
    Term =.. [Space, Name | Args], 
    append(Decls2, [Term], Out).
