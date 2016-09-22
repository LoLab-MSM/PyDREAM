# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 19:56:12 2014

@author: Erin
"""

from pysb import Model, Monomer, Parameter, Initial, Rule, Observable
from pysb.macros import bind, bind_complex, catalyze

Model()

#Define individual species in model
Monomer('COX2', ['allo', 'cat']) #Cyclooxygenase-2 enzyme
Monomer('AG', ['b']) #2-arachidonoylglycerol, a substrate of COX2
Monomer('AA', ['b']) #arachidonic acid, a substrate of COX2
Monomer('PG') #Prostaglandin, product of COX2 turnover of AA
Monomer('PGG') #Prostaglandin glycerol, product of COX2 turnover of 2-AG

#Initial starting concentrations in micromolar
Parameter('COX2_0', 15e-3)
Parameter('AG_0', 16)
Parameter('AA_0', 16)
Parameter('PG_0', 0)
Parameter('PGG_0', 0)

Initial(COX2(allo=None, cat=None), COX2_0)
Initial(AG(b=None), AG_0)
Initial(AA(b=None), AA_0)
Initial(PG(), PG_0)
Initial(PGG(), PGG_0)

#All kf parameters are in units of inverse microM*s
#All kr parameters are in units of inverse s
#All kcat parameters are in units of inverse s
#the forward reaction is association; the reverse is disassociation

#Rates for AA and COX2 interactions at catalytic site
Parameter('kf_AA_cat1', 1000.0)
Parameter('kr_AA_cat1', 830) 
Parameter('kcat_AA1', 1.3) 
Parameter('kf_AA_cat2', 1.0e-3)
Parameter('kr_AA_cat2', 3.3e-6)
Parameter('kcat_AA2', 2.3)
Parameter('kf_AA_cat3', 1.0e-3)
Parameter('kr_AA_cat3', 8.3e-6)
Parameter('kcat_AA3', 1.3)

#Rates for 2-AG and COX2 interactions at catalytic site
Parameter('kf_AG_cat1', 1000.0)
Parameter('kr_AG_cat1', 760.0) 
Parameter('kcat_AG1', 1.2) 
Parameter('kf_AG_cat2', 1.0e-3)
Parameter('kr_AG_cat2', 4.8e-4)
Parameter('kf_AG_cat3', 1.0e-3)
Parameter('kr_AG_cat3', 1.9e-6)
Parameter('kcat_AG3', 0.21)

#Rates for AA and COX2 interactions at allosteric site
Parameter('kf_AA_allo1', 1000.0)
Parameter('kr_AA_allo1', 1.0e5)
Parameter('kf_AA_allo2', 1000.0)
Parameter('kr_AA_allo2', 1000.0)
Parameter('kf_AA_allo3', 1000.0)
Parameter('kr_AA_allo3', 250.0)

#Rates for 2-AG and COX2 interactions at allosteric site
Parameter('kf_AG_allo1', 1000.0)
Parameter('kr_AG_allo1', 1.0e5)
Parameter('kf_AG_allo2', 1000.0)
Parameter('kr_AG_allo2', 400.0)
Parameter('kf_AG_allo3', 1000.0)
Parameter('kr_AG_allo3', 63000.0) 

#Defining allowed reaction rules

catalyze(COX2(allo=None), 'cat', AA(), 'b', PG(), [kf_AA_cat1, kr_AA_cat1, kcat_AA1])

bind_complex(COX2(allo=1) % AG(b=1), 'cat', AA(), 'b', [kf_AA_cat2, kr_AA_cat2])

Rule('kcat_AA_2',
     COX2(allo=1, cat=2) % AG(b=1) % AA(b=2) >> COX2(allo=1, cat=None) % AG(b=1) + PG(),
    kcat_AA2)

bind_complex(COX2(allo=1) % AA(b=1), 'cat', AA(), 'b', [kf_AA_cat3, kr_AA_cat3])

Rule('kcat_AA_3',
     COX2(allo=1, cat=2) % AA(b=1) % AA(b=2) >> COX2(allo=1, cat=None) % AA(b=1) + PG(),
    kcat_AA3)

catalyze(COX2(allo=None), 'cat', AG(), 'b', PGG(), [kf_AG_cat1, kr_AG_cat1, kcat_AG1])

bind_complex(COX2(allo=1) % AG(b=1), 'cat', AG(), 'b', [kf_AG_cat2, kr_AG_cat2])

bind_complex(COX2(allo=1) % AA(b=1), 'cat', AG(), 'b', [kf_AG_cat3, kr_AG_cat3])

Rule('kcat_AG_3',
     COX2(allo=1, cat=2) % AA(b=1) % AG(b=2) >> COX2(allo=1, cat=None) % AA(b=1) + PGG(),
    kcat_AG3)

bind(COX2(cat=None), 'allo', AA(), 'b', [kf_AA_allo1, kr_AA_allo1])

Rule('bind_COX2AA_AA_allo',
     COX2(cat=1, allo=None) % AA(b=1) + AA(b=None) <> COX2(cat=1, allo=2) % AA(b=1) % AA(b=2),
    kf_AA_allo2, kr_AA_allo2)

Rule('bind_COX2AG_AA_allo',
     COX2(cat=1, allo=None) % AG(b=1) + AA(b=None) <> COX2(cat=1, allo=2) % AG(b=1) % AA(b=2),
    kf_AA_allo3, kr_AA_allo3)

bind(COX2(cat=None), 'allo', AG(), 'b', [kf_AG_allo1, kr_AG_allo1])

Rule('bind_COX2AA_AG_allo',
     COX2(cat=1, allo=None) % AA(b=1) + AG(b=None) <> COX2(cat=1, allo=2) % AA(b=1) % AG(b=2),
    kf_AG_allo2, kr_AG_allo2)

Rule('bind_COX2AG_AG_allo',
     COX2(cat=1, allo=None) % AG(b=1) + AG(b=None) <> COX2(cat=1, allo=2) % AG(b=1) % AG(b=2),
    kf_AG_allo3, kr_AG_allo3)

Observable('obsPG', PG())
Observable('obsPGG', PGG())
Observable('obsAA', AA())
Observable('obsAG', AG())
Observable('obsAAallo', COX2(allo=1, cat=None) % AA(b=1))
Observable('obsAAcat', COX2(cat=1, allo=None) % AA(b=1))
Observable('obsAAboth', COX2(cat=1, allo=2) % AA(b=1) % AA(b=2))
Observable('obsAGallo', COX2(allo=1, cat=None) % AG(b=1))
Observable('obsAGcat', COX2(cat=1, allo=None) % AG(b=1))
Observable('obsAGboth', COX2(cat=1, allo=2) % AG(b=1) % AG(b=2))
Observable('obsAAcatAGallo', COX2(cat=1, allo=2) % AA(b=1) % AG(b=2))
Observable('obsAGcatAAallo', COX2(cat=1, allo=2) % AG(b=1) % AA(b=2))



