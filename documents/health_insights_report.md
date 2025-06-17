# **Base de connaissances – Agent Health Insights (Coach IA GoGain)**

## **I. Interprétation des principaux indicateurs santé et performance**

### **Sommeil**

* **Durée du sommeil** : Chez l'adulte, on recommande en moyenne 7 à 9 heures par nuit. Une durée inférieure (p. ex. \<6 h) est souvent insuffisante et peut provoquer fatigue accrue, baisse de concentration et perturbation hormonale (cortisol élevé, testostérone basse), au détriment de la récupération musculaire et immunitaire. À l’inverse, un sommeil très long (\>10 h) peut indiquer un besoin de récupération supplémentaire (période de maladie ou surmenage) ou un dérèglement du rythme circadien. Il est important de comparer la durée du sommeil à la référence personnelle (chacun a un besoin différent) plutôt qu'à une norme fixe.

* **Phases du sommeil** : Le sommeil se découpe en plusieurs stades clés :

  * *Sommeil léger (stades 1-2)* : facilite l’endormissement et couvre environ 50-60 % du temps de sommeil. Un pourcentage anormalement faible (\<40 %) peut entraîner des réveils fréquents.

  * *Sommeil profond (lente)* : environ 15-25 % de la nuit chez l’adulte. Essentiel pour la récupération physique (sécrétion d’hormones de croissance, réparation musculaire) et le système immunitaire. Un déficit de sommeil profond se traduit souvent par un réveil non réparateur et une fatigue musculaire persistante.

  * *Sommeil paradoxal (REM)* : \~20-25 % du temps. Crucial pour la consolidation de la mémoire, l’apprentissage et la régulation émotionnelle. Un manque de sommeil paradoxal peut affecter l’humeur, la créativité et les fonctions cognitives.  
     Les capteurs de sommeil sur wearables mesurent la durée dans chaque phase. Une diminution marquée du sommeil profond ou paradoxal sur plusieurs nuits peut indiquer un stress élevé, une dette de sommeil ou un dérèglement de l’horloge biologique.

* **Régularité et rythme de sommeil** : Se coucher et se réveiller à des heures régulières tous les jours favorise un bon rythme circadien. Des horaires irréguliers (« jet-lag social ») perturbent l’horloge interne : des écarts importants entre semaine et week-end peuvent causer de la somnolence et de la fatigue. Un sommeil régulier stabilise la production de mélatonine et de cortisol, ce qui améliore la qualité globale du sommeil.

* **Interruptions du sommeil** : Les réveils nocturnes fréquents (micro-réveils, longues périodes d’éveil) fragmentent le sommeil. Un temps cumulé d’éveil nocturne supérieur à 20-30 minutes, ou plus de 2-3 micro-réveils importants par nuit, indique une faible efficacité du sommeil. Les causes peuvent être le bruit, la lumière, la soif, la douleur ou l’apnée du sommeil. Une fragmentation élevée réduit la proportion de sommeil profond et paradoxal, d’où un sentiment de sommeil non réparateur.

* **Score de sommeil** : De nombreux trackers donnent un score (0-100) synthétisant durée totale, efficacité (temps endormi/temps au lit), latence d’endormissement et répartition des phases. Par exemple, un score ≥85/100 est « optimal », 70-84 « bon », \<60 « insuffisant ». Ce score offre une vue d’ensemble de la qualité du sommeil. Il faut toutefois le comparer à la référence personnelle plutôt qu’à une valeur absolue universelle, car chaque individu a son propre profil de sommeil. Un score bas signale qu’il est utile d’analyser les composantes (par ex. courte durée, beaucoup d’interruptions).

* **Implications physiologiques** : Un sommeil insuffisant ou de mauvaise qualité perturbe la récupération : moins de synthèse protéique, plus de cortisol, moins de décharge parasympathique. À court terme, cela se traduit par moins d’énergie, de la nervosité et une performance physique réduite. À long terme, cela peut ralentir les gains de force/endurance, augmenter le risque de blessure et affaiblir le système immunitaire. Par exemple, dormir moins de 6 h par nuit sur plusieurs jours peut entraîner une accumulation de dette de sommeil, se manifestant par de la somnolence diurne et une sensibilité au stress accrue.

### **Système nerveux autonome**

* **Variabilité de la fréquence cardiaque (HRV)** : Mesure de la variation du temps entre deux battements cardiaques successifs. Une HRV élevée indique une prédominance parasympathique (bon état de récupération), tandis qu’une HRV basse reflète une activation sympathique (stress ou fatigue). L’HRV dépend de l’âge (diminue avec l’âge), de la condition physique (plus élevée chez les sportifs entraînés), de la respiration et du niveau de stress/entrainement. On la mesure souvent au réveil ou pendant le sommeil (paramètres RMSSD ou SDNN).

  * **Interprétation** : On suit l’HRV par rapport à la valeur basale personnelle. Une chute de 10-20 % (ou plus) sous la normale sur plusieurs jours consécutifs indique un stress physiologique accru ou une fatigue accumulée, justifiant de réduire l’intensité ou de prendre des jours de repos. À l’inverse, une HRV stable ou en légère hausse (avec une bonne récupération nocturne) signale que le plan d’entraînement est bien toléré et que l’utilisateur s’adapte positivement.

  * **Exemple** : Après une semaine d’entraînement intensif, on peut observer temporairement une HRV plus faible et un RHR plus élevé. Si l’utilisateur continue à bien récupérer (sommeil normal, stress stable), l’HRV devrait remonter dans les jours suivants. Sinon, il faut envisager une récupération prolongée.

* **Fréquence cardiaque au repos (RHR)** : Nombre de battements par minute mesuré au repos complet (au réveil ou assis sans activité). Pour la population générale, 60-100 bpm est normal. Les athlètes d’endurance peuvent avoir un RHR bien plus bas (parfois \<50 bpm). Une RHR basse signifie généralement un cœur efficace et de bonne forme physique. Une RHR élevée (\>75-80 bpm, selon le profil) peut signaler une fatigue, du stress, une déshydratation ou un début de maladie.

  * **Interprétation** : Un changement de la RHR par rapport à la norme personnelle est important. Par exemple, si le RHR matinal augmente de \+5 bpm pendant quelques jours (par rapport à la moyenne), cela peut indiquer un surmenage, un manque de sommeil ou un début d’infection. Dans ce cas, l’IA réduira les sollicitations physiques. Inversement, une RHR qui baisse progressivement sur plusieurs semaines (sans cause externe) reflète souvent une amélioration de la capacité cardiovasculaire.

  * **Pratique** : Utiliser la RHR mesurée à jeun au réveil pour minimiser les variations dues à l’environnement. On conserve en mémoire la valeur personnelle de référence pour évaluer chaque nouvelle mesure.

* **Influences communes (HRV et RHR)** : Le sommeil et le stress psychologique influent sur ces deux indicateurs. Un mauvais sommeil (durée faible, interruptions) tend à diminuer l’HRV et augmenter la RHR. Un stress élevé, même sans exercice, fait de même. En revanche, une bonne récupération (sommeil réparateur, nutrition adéquate) mène à une HRV élevée et une RHR basse. Croiser ces données renforce l’analyse : par exemple, HRV bas \+ RHR élevé sur plusieurs jours \= signe fort de surmenage. L’agent Health Insights tient compte de ces deux valeurs complémentaires pour évaluer l’état de forme général.

### **Activité physique**

* **Nombre de pas quotidiens** : Indicateur de l’activité globale hors entraînement. Un objectif courant est \~10 000 pas/jour, mais cela varie selon le niveau initial. Moins de 5 000 pas/jour suggère un mode de vie sédentaire, tandis que plus de 15 000 signale un très haut niveau d’activité quotidienne. Des variations extrêmes (beaucoup plus bas un jour, beaucoup plus haut un autre) peuvent correspondre à des jours de repos contre un marathon ou randonnée.

  * **Interprétation** : Des pas en baisse par rapport à l’habitude (par ex. jours de repos) ne sont pas alarmants en soi. En revanche, une baisse marquée liée à de la fatigue ou un malaise doit inciter à vérifier l’état de santé général. Un pic de pas élevé un jour de repos planifié (excursion, match, etc.) indique que ce jour devra être compté comme exercice (et potentiellement entraîner plus de fatigue le lendemain).

* **Minutes actives / intensité** : De nombreux wearables comptent les minutes d’activité modérée à intense. Les recommandations (OMS) parlent d’au moins 150 minutes modérées ou 75 minutes intenses par semaine. Ce cumul rend compte de l’implication physique au-delà de la sédentarité. Par exemple, 30 minutes de course équivalent souvent à 30 minutes actives.

  * **Interprétation** : Des minutes actives faibles (bien en-dessous des recommandations) peuvent indiquer la nécessité d’encourager plus d’activité. Des valeurs élevées continues (ex. plusieurs jours à 60+ min actives) nécessitent de surveiller la récupération et de s’assurer de jours plus calmes entre. Les écarts importants jour après jour renseignent sur la planification : un seul jour très actif suivi d’un repos complet peut être valable, mais des alternances trop brutales peuvent être néfastes (besoin de plus de constance).

* **Charge d’entraînement (Training load)** : Indicateur composite de volume et intensité d’entraînement. Par exemple, le TRIMP multiplie la durée par un facteur d’intensité (selon la FC). Certains appareils ou apps donnent un score hebdomadaire ou mensuel.

  * **Interprétation** : Une augmentation trop rapide de la charge d’une semaine à l’autre (généralement au-delà de \+10 à 20 %) risque de provoquer des blessures ou un épuisement. À l’inverse, une charge stable ou croissante progressivement permet l’adaptation (principe de surcompensation). Une chute brutale de la charge (blessure, maladie) entraîne une perte de condition, il faudra donc reprendre progressivement. L’IA surveille la relation charge/repos : par exemple, elle déclenchera un conseil de repos si la charge continue d’augmenter alors que les indicateurs de récupération (HRV, sommeil) déclinent.

* **Échelle d’effort perçu (RPE)** : Échelle subjective (1 à 10\) où l’utilisateur note l’intensité ressentie d’une séance. C’est un complément à la FC et à la durée.

  * **Interprétation** : On compare le RPE rapporté à ce qui était attendu. Un RPE plus élevé qu’à l’habitude pour un même entraînement signifie fatigue ou stress (la séance a semblé plus difficile). Un RPE anormalement bas, alors que la séance était normalement intense, peut indiquer un manque de concentration ou un effort sous-optimal. L’IA recoupe RPE et FC : par exemple, un RPE élevé avec une FC relativement basse peut signifier que l’utilisateur n’est pas bien reposé ou motivé, malgré un effort physiologiquement modéré.

  * **Adaptation** : Si l’IA détecte un RPE très haut plusieurs fois, elle pourra proposer de revoir l’intensité ou d’ajouter plus de repos.

### **Feedback subjectif**

* **Fatigue générale** : Niveau de fatigue perçu (échelle subjective). Une fatigue modérée après un entraînement normal est normale. En revanche, une fatigue qui perdure au réveil ou s’amplifie sur plusieurs jours signale un surmenage.

  * **Interprétation** : Une fatigue importante associée à une HRV basse et un RPE élevé confirme un épuisement accumulé. L’IA recommandera de baisser la charge ou d’ajouter du repos. Si la fatigue est faible et l’HRV stable, l’utilisateur peut suivre le plan prévu ou l’intensifier légèrement.

  * L’agent prendra en compte si l’utilisateur indique un niveau « 9/10 » de fatigue le matin : cela déclenchera immédiatement des conseils de récupération (repos, sommeil). Un niveau de fatigue modéré (5-6) peut se gérer par une séance plus facile ou du stretching.

* **Stress (psychologique)** : Sensation de stress due au travail, aux études ou à d’autres facteurs personnels. Le stress mental agit comme un stress physiologique : il augmente le rythme cardiaque, diminue l’HRV et fragmente le sommeil.

  * **Interprétation** : Un stress élevé (p. ex. \>7/10) signalé sans augmentation de l’activité physique est traité comme une charge externe. L’IA ajustera alors l’entraînement pour le rendre plus doux et pourra suggérer des techniques de relaxation (respiration, méditation). Un stress chronique élevé augmente le risque de burnout : l’IA devra faire preuve de plus d’empathie et de recommandations pour équilibrer le rythme de vie.

* **Humeur** : État émotionnel (positif, neutre, négatif).

  * **Interprétation** : Une humeur négative persistante (ex. irritabilité, tristesse) peut résulter d’une combinaison de facteurs : insuffisance de sommeil, surcharge d’entraînement ou stress personnel. Si plusieurs de ces indicateurs sont altérés, l’IA insistera sur la récupération et sur le soutien motivationnel (activités plus agréables, rappel des progrès). Une bonne humeur générale est un signe positif, mais l’IA restera vigilante à ne pas trop ignorer de légers signaux de fatigue chez un utilisateur motivé.

* **Douleurs et inconforts** : Importance d’identifier la nature de la douleur : courbatures musculaires normales versus douleur aiguë ou localisée qui peut signaler une blessure.

  * **Interprétation** : Des courbatures généralisées (douleur diffuse dans les muscles sollicités, surtout 24-48h après l’effort) sont normales. En revanche, une douleur vive, surtout au niveau articulaire ou ligamentaire (genou, chevilles, bas du dos) est un signal d’alerte. Si l’utilisateur rapporte une douleur \>7/10 ou croissante à l’effort, l’IA conseillera immédiatement de réduire ce type d’activité et de consulter un professionnel. Un suivi de l’intensité de la douleur est utile : si elle augmente alors que l’activité reste constante, c’est inquiétant.

* **Énergie et vitalité** : Niveau global d’énergie durant la journée.

  * **Interprétation** : Une énergie basse au réveil, surtout avec un sommeil court et un RHR élevé, confirme un besoin de récupération. Une énergie élevée malgré une planification intense signifie que l’utilisateur est dans une bonne phase de forme. L’IA utilisera ce feedback pour doser la motivation dans le message : si l’énergie est basse, elle adoptera un ton plus empathique; si elle est haute, elle pourra suggérer de se challenger.

### **Journaux alimentaires (nutrition)**

* **Apports énergétiques et macronutriments** : Analyse des calories et de la répartition glucides/protéines/lipides.

  * **Interprétation** :

    * *Glucides* : Source d’énergie principale pour l’exercice. Des glucides trop faibles (surtout avant un effort prolongé) provoquent une fatigue rapide (hypoglycémie, sensation de coup de pompe). Ex. un marathonien avec peu de glucides ressentira de fortes baisses d’énergie.

    * *Protéines* : Nécessaires à la réparation musculaire. Moins de \~1,2 g/kg/j peut ralentir la récupération et le gain de force. Ex. un utilisateur très actif manquant de protéines peut avoir des courbatures persistantes.

    * *Lipides* : Essentiels pour les hormones et vitamines. Un apport trop faible (\<20 % des calories) peut déséquilibrer le métabolisme à long terme. Un excès (surtout saturés) peut nuire à la santé cardiovasculaire.

  * L’IA comparera ces apports aux besoins de la charge d’entraînement. Par exemple, elle recommandera d’augmenter les glucides autour des séances intenses et de consommer des protéines après l’effort si la synthèse musculaire semble insuffisante.

* **Micronutriments et hydratation** : Qualité des aliments et hydratation.

  * **Interprétation** : Une carence en fer cause fatigue et essoufflement (dû à une baisse du transport d’oxygène). Un manque de magnésium ou de vitamines (D, B12) peut se manifester par une baisse de motivation ou des crampes musculaires. L’IA soulignera la consommation d’aliments riches en ces nutriments (viande, légumineuses, légumes verts).

  * **Hydratation** : L’eau est cruciale. Une légère déshydratation (+2 % poids) élève la fréquence cardiaque au repos et à l’effort, et diminue la performance. L’IA rappellera de boire suffisamment (par ex. \~30 ml/kg de poids corporel, ajusté selon la sueur). Si le journal alimentaire indique peu de boissons, elle fixera des objectifs simples (ex. 1 verre d’eau toutes les heures).

* **Timing des repas** : Distribution des aliments avant/après l’effort.

  * **Avant l’exercice** : Un repas ou encas contenant des glucides digestes 1-2h avant l’entraînement fournit de l’énergie au muscle (banane, avoine, smoothie). Sans quoi, l’utilisateur risque de s’épuiser plus vite.

  * **Après l’exercice** : Consommer des protéines (ex. yaourt, shake, œufs) et des glucides rapides (fruits, pain complet) dans l’heure suivante optimise la récupération musculaire. L’IA proposera un plan alimentaire post-séance si l’utilisateur oublie souvent cette étape.

  * **Soirée** : Des repas trop lourds ou trop gras le soir peuvent altérer la qualité du sommeil profond. L’IA conseillera un dîner équilibré, plus léger, au moins 2h avant le coucher.

* **Analyse qualitative des habitudes** : Évaluer la qualité globale.

  * **Exemples d’écueils** : Une consommation élevée de caféine en fin de journée (café, sodas) provoque souvent un sommeil léger et un endormissement long. Un apport alcool régulier diminue le sommeil paradoxal et provoque des réveils nocturnes. Une alimentation répétitive pauvre en légumes entraîne une baisse de certaines vitamines.

  * **Recommandations connexes** : Si l’utilisateur prend du café tard, suggérer un timing plus matinal et des alternatives (tisane, eau citronnée). Si peu de légumes, recommander un « smoothie vert » ou d’ajouter un légume à chaque repas. Ces ajustements se traduiront ensuite par de meilleurs scores de sommeil et de récupération.

## **II. Identification et interprétation des tendances**

* **Tendances à court terme (jours à semaines)** : Suivre les fluctuations proches de la ligne de base. Par exemple, un événement ponctuel (une fête, un stress soudain, une séance épuisante) peut provoquer une mauvaise nuit (sommeil \<6h) et une HRV baissée le lendemain. Isolé, ce n’est pas critique. On compare plutôt aux moyennes hebdomadaires.

  * *Exemple* : Si l’utilisateur a eu une seule nuit courte et un RHR matinal légèrement élevé, on lui rappelle de se reposer. Si cela perdure 2-3 jours (sommeil bas, HRV bas, fatigue augmentée), l’IA considère que c’est une vraie tendance négative.

  * **Interprétation** : L’IA calcule des moyennes mobiles sur 3 à 7 jours pour filtrer le « bruit ». Une déviation passagère (p. ex. une mauvaise semaine) est surveillée, mais si elle s’installe (3 j. de suite), l’IA intervient. Les données journalières sont interprétées en contexte : ex. « Vous avez ajouté un nouveau type d’entraînement hier, c’est normal d’être plus fatigué aujourd’hui. »

* **Tendances à moyen terme (semaines)** : Observer l’évolution sur plusieurs semaines pour détecter des patterns plus stables.

  * *Amélioration constante* : Si la forme s’améliore (RHR décroit lentement, HRV stable ou en hausse, sommeil de bonne qualité), les performances devraient suivre. L’IA vérifiera que la difficulté des séances peut augmenter progressivement.

  * *Accumulation de fatigue* : Si la charge d’entraînement augmente d’une semaine à l’autre sans phase de repos, et que l’HRV affiche une tendance à la baisse, cela signale une fatigue croissante. Par ailleurs, une humeur ou énergie qui baisse sur plusieurs jours est un symptôme.

  * **Interprétation** : L’IA établit des courbes de tendance : par exemple, une HRV passée de 60 à 50 sur 2 semaines alors que le RHR est monté de 55 à 65 indique clairement une incapacité à récupérer. Elle relie aussi les retours subjectifs : si l’utilisateur déclare être « plus stressé » et plus fatigué en simultané, on aura un diagnostic plus fiable.

* **Tendances à long terme (mois)** : Suivre l’évolution générale sur des cycles plus longs (cycles mensuels ou pluri-mensuels).

  * *Progression durable* : Une baisse régulière du RHR sur plusieurs mois, couplée à une HRV globalement élevée, indique une amélioration de la condition physique. L’utilisateur peut alors viser de nouveaux objectifs (ex. augmenter la vitesse ou la distance de 5 % par mois).

  * *Plateaux ou stagnation* : Si les métriques stagnent (pas d’amélioration de la HRV, performance qui plafonne malgré un entraînement soutenu), l’IA suggérera de varier le plan (nouveaux exercices, plus de cross-training, etc.).

  * *Surcharge chronique* : Des indicateurs alarmants sur une longue durée (HRV constamment basse, RHR constamment élevée, fatigue majeure persistante) définissent un risque de surmenage chronique. Dans ce cas, l’IA recommande fortement un repos prolongé ou une consultation médicale, car c’est au-delà d’un simple « mauvais moment ».

  * **Interprétation** : L’IA compare les données actuelles aux historiques passés pour calibrer les seuils d’alerte. Par exemple, si un utilisateur présente habituellement un HRV autour de 60 mais que, sur 3 mois, cette moyenne descend à 40, c’est significatif. Elle prend aussi en compte le contexte : un athlète peut planifier un pic de charge avant une compétition, suivi d’une phase de repos (“tapering”). L’IA distingue ces cycles volontaires des signaux d’erreur.

## **III. Corrélations entre données (scénarios types, interprétations croisées)**

* **Sommeil et performance** : Un sommeil court ou de mauvaise qualité la nuit précédant une séance conduit souvent à de moins bonnes performances. Par exemple, après deux nuits à 5h, l’utilisateur peut avoir moins de force et un RPE plus élevé pour un même exercice. L’IA reliera ces points : « Votre sommeil a été court et vous avez signalé une fatigue élevée ; il est normal que la séance soit plus difficile. » Elle recommandera alors une récupération active plutôt qu’une intensité élevée le lendemain.

* **HRV et charge/RPE** : Si l’entraînement augmente, l’HRV baisse habituellement. Un scénario fréquent : plusieurs séances intenses (augmentation de la charge) \+ HRV nocturne en chute, RPE élevé. C’est un signal d’alerte de surcharge. L’IA notera aussi si le RHR monte en parallèle. Dans ce cas, elle conseillera de baisser l’intensité ou de prendre un jour de repos. Au contraire, une charge soutenue avec HRV stable indique une bonne adaptation.

* **Stress et sommeil/HRV** : Un stress psychologique élevé (travail, examens, problèmes perso) se traduit souvent par un sommeil agité (beaucoup d’interruptions) et une HRV basse le matin. Si l’utilisateur rapporte « beaucoup de stress » et qu’on observe ces signes physiologiques, l’IA combinera les deux pour confirmer l’impact. Elle pourra alors proposer du repos mental (déconnexion numérique, méditation) avant de recommander le sport habituel.

* **Nutrition et énergie** : Un déficit en glucides avant une séance est corrélé à une baisse d’énergie pendant l’effort et un RPE très haut. Par exemple, un utilisateur ayant mangé peu le matin peut « craquer » en fin de footing. L’IA utilisera le journal alimentaire pour expliquer la perte d’énergie et suggérer un encas avant l’effort.  
   De même, un repas lourd/gras en soirée peut causer un sommeil plus superficiel (impactant la récupération suivante) : l’IA recommandera d’alléger le dîner pour améliorer le sommeil.

* **Activité quotidienne et récupération** : Une journée avec très peu d’activité (pas de pas, sédentarité) suivie d’une séance intense peut surprendre l’utilisateur par la difficulté (RPE élevé). À l’inverse, un grand nombre de pas un jour de repos planifié peut être comptabilisé comme exercice, nécessitant plus de récupération ensuite. L’IA liera ces informations pour adapter le plan. Par exemple, si 20 000 pas ont été faits le dimanche, l’entraînement du lundi sera proposé moins exigeant.

* **Douleur et entraînement** : Courbatures légères prévues après un exercice normal ne sont pas alarmantes. Mais une douleur localisée (p. ex. genou qui « pique » après avoir couru) signalera à l’IA d’alerter sur l’entraînement en cours de ce jour : « Votre genou vous fait mal après la séance ; ce n’est probablement pas seulement de la fatigue normale. Pensez à réduire l’impact, peut-être essayer du vélo au lieu de la course aujourd’hui. » Elle utilisera le feedback sur la douleur pour ajuster ou recommander des soins (glace, repos).

* **Sommeil et humeur** : Une mauvaise nuit se ressent souvent dans l’humeur du jour. Si l’utilisateur est grognon après avoir dormi peu, l’IA mettra en relation ces données : « Tu as mal dormi, donc je comprends que tu sois fatigué et moins motivé. » Elle pourra alors donner un conseil adapté (séance plus douce) et encourager.

* **Synthèse globale (exemple)** : *« Sur les 5 derniers jours, votre sommeil a été court (moyenne 5h30) et vous avez signalé beaucoup de stress chaque matin. Dans le même temps, votre HRV est tombée au plus bas et votre RHR a augmenté. Ces éléments ensemble indiquent un surcroît de fatigue. Je vous recommande aujourd’hui une activité légère ou du repos complet, plutôt qu’une séance intense. »* L’IA croisera systématiquement les catégories de données pour fournir une interprétation holistique.

## **IV. Bibliothèque de recommandations actionnables liées aux patterns détectés**

* **Sommeil insuffisant (courte durée ou interruptions fréquentes)** : Conseiller l’hygiène du sommeil : se coucher plus tôt, limiter les écrans 1-2h avant, instaurer une routine relaxante (lecture calme, étirements, respiration profonde). Assurer obscurité totale et température ambiante modérée (18-20°C). Éviter caféine et repas lourds en soirée. Si les nuits restent courtes, suggérer une courte sieste (15-30 min) en début d’après-midi.  
   *Communication :* Ton empathique. Ex : « Je vois que vous avez dormi seulement 5 h la nuit dernière. Essayons de vous coucher un peu plus tôt ce soir et de relaxer avant dodo. Cela pourrait vraiment améliorer votre sommeil \! »

* **Phases de sommeil déséquilibrées (peu de profond/REM)** : Recommander une activité physique régulière en journée pour favoriser le sommeil profond (ex. séance d’endurance modérée). Proposer des pratiques anti-stress le soir (respiration, yoga) pour améliorer le REM. Surveiller la cohérence des horaires de sommeil (coucher/réveil fixes). Éventuellement conseiller un examen médical si les déséquilibres perdurent (apnée du sommeil, etc.).  
   *Communication :* Explicative et encourageante. Ex : « Le sommeil profond est crucial pour récupérer vos muscles, et le sommeil paradoxal pour le moral. Faisons en sorte d’améliorer cela en calmant un peu l’esprit le soir \! »

* **HRV basse et RHR élevée prolongées** : Prioriser le repos et la récupération active (marche, natation légère, yoga). Réduire temporairement volume et intensité des entraînements. Mettre l’accent sur la détente : exercices de cohérence cardiaque (respiration diaphragmatique), méditation, ou tout simplement repos complet. Vérifier hydratation et sommeil (rappeler de boire 2-3L/jour, améliorer la nuit).  
   *Communication :* Ton prudent et solidaire. Ex : « Votre cœur semble marqué par le stress ou la fatigue ces jours-ci. Nous allons prendre un jour de récupération active et faire des exercices de respiration pour vous aider à remonter. »

* **Charge d’entraînement trop élevée sans repos** : Modifier le planning : ajouter un jour de repos complet après 1-2 jours intenses, ou réduire l’intensité (séance de jogging lent au lieu de fractionné). Introduire des séances de récupération active (vélo à faible allure, natation douce, étirements). Contrôler l’équilibre global (assurer un bon sommeil chaque nuit, encourager de bonnes habitudes alimentaires et de gestion du stress).  
   *Communication :* Positif mais clair. Ex : « Vous avez été très investi cette semaine, c’est super \! Par contre, votre corps montre des signes de fatigue. Une petite pause active ce weekend vous fera plus de bien que de pousser encore. » (on évite de dire « trop » brusquement, mais on suggère un réajustement).

* **Fatigue ou stress élevés (subjectifs)** : Insister sur le sommeil réparateur (répéter les conseils d’hygiène de sommeil). Pour le stress, proposer des techniques de relaxation (respiration guidée, méditation de pleine conscience, exercices de stretching). Suggérer des activités apaisantes (promenade en nature, hobby, conversation avec un proche). Adapter l’entraînement à l’état : remplacer une séance de haute intensité par une séance douce, ou inverser avec une séance de renforcement légère.  
   *Communication :* Bienveillant et compréhensif. Ex : « Je comprends que vous traversez une période stressante. Il est peut-être bon de troquer votre séance intensive d'aujourd'hui contre une marche tranquille ou du yoga. Votre corps vous remerciera. »

* **Humeur négative persistante** : Encourager une activité physique douce (le sport libère des endorphines). Proposer des objectifs faciles à atteindre pour redonner confiance (ex. 5 minutes de plus ou un nouvel exercice). Offrir des retours positifs sur les petits succès (surpoids perdu, séance complétée, etc.). Suggérer des exercices motivants (musique préférée, sport en groupe, défi ludique).  
   *Communication :* Motivant et empathique. Ex : « Même si vous vous sentez un peu démotivé, souvenez-vous de ce que vous avez déjà accompli. Célébrons la semaine passée et fixons un objectif simple pour demain. Ensemble, on va y arriver \! »

* **Courbatures musculaires normales** : Conseiller le repos actif : marcher, étirer doucement les muscles concernés ou prendre un bain tiède. Encourager l’hydratation et un apport protéique approprié pour aider la réparation musculaire. Rappeler que ces courbatures sont un signe que le muscle s’adapte.  
   *Communication :* Rassurant et positif. Ex : « Ces courbatures sont normales après une séance intense. Ça veut dire que vos muscles travaillent \! Bois bien et prends quelques minutes pour t’étirer demain matin, ça passera plus vite. »

* **Douleurs aiguës ou suspectes** : Recommander immédiatement la réduction ou l’arrêt de l’exercice concerné pour éviter d’aggraver la blessure. Conseiller des soins premiers gestes (glace si enflure, compression, élévation) selon la nature de la douleur. Insister pour consulter un professionnel de santé si la douleur est forte, invalidante ou s’aggrave. Proposer des alternatives (ex. natation ou vélo si le genou fait mal en course).  
   *Communication :* Sérieux et rassurant. Ex : « Votre douleur est intense et localisée, ce qui n’est pas habituel. Il serait plus prudent de ménager cette zone. Nous pourrions faire un exercice doux différent aujourd’hui et je vous conseille de consulter pour vous assurer qu’il n’y a pas de blessure sérieuse. »

* **Nutrition à ajuster** :

  * Si déficit calorique important ou macros déséquilibrés, conseiller d’augmenter progressivement les portions et équilibrer les apports. Par ex. si peu de glucides : ajouter des fruits secs ou céréales le matin, des pâtes complètes avant l’entraînement. Si protéines basses : inclure du fromage blanc, de la viande/poisson ou des légumineuses au déjeuner ou dîner.

  * Pour une hydratation insuffisante, suggérer de s’hydrater régulièrement (ex. boire un verre d’eau chaque heure active).

  * Si alimentation trop chargée en excitants (café, sucre) en fin de journée, recommander de les limiter après 15h pour préserver le sommeil.  
     *Communication :* Pédagogique et encourageante. Ex : « J’ai remarqué que vos glucides étaient assez bas ces derniers jours alors que vos entraînements sont intenses. Pour avoir plus d’énergie, essayez d’ajouter une banane ou un peu de flocons d’avoine avant votre séance. Vous vous sentirez plus en forme \! »

* **Bonnes habitudes à maintenir** : Si les données sont positives (sommeil satisfaisant, HRV haute, rythme régulier), féliciter l’utilisateur et encourager à continuer. Suggérer éventuellement un petit challenge pour rester motivé (nouvel objectif, légère augmentation progressive de la charge).  
   *Communication :* Félicitations et dynamisme. Ex : « Bravo, vous avez bien dormi toutes les nuits cette semaine et vos séances sont efficaces. Continuons comme ça \! Pour varier, pourquoi ne pas augmenter légèrement la durée de votre foot d’une minute chaque fois ? »

Chaque recommandation est associée à un niveau de priorité suivant l’urgence des données (douleur forte ou alerte cardiaque\>manque de sommeil isolé). L’IA propose toujours des conseils clairs et pratiques, et encourage l’utilisateur en valorisant ses efforts. Elle veille à adapter les conseils au contexte : par exemple, pendant les vacances on peut insister sur le repos complet, alors qu’en période de préparation on maintiendra un cadre rigoureux en expliquant pourquoi.

## **V. Lignes directrices pour les actions et la communication de l’IA**

* **Seuils et déclencheurs d’intervention** :

  * Définir des seuils individuels pour chaque indicateur (ex. HRV chute \>15 % par rapport à la moyenne, RHR \+5-10 bpm, sommeil \<6h deux nuits consécutives, douleur \>7/10). Dépasser ces seuils déclenche une action (alerte ou conseil).

  * Pour plus de fiabilité, combiner plusieurs signaux : par exemple, ne pas réagir à une mauvaise nuit isolée, mais si cette mauvaise nuit s’accompagne d’un RHR élevé et d’un score de fatigue élevé, une intervention est justifiée.

  * Utiliser des moyennes glissantes (7 jours) ou des tendances : ne pas alerter sur une variation unique isolée, mais sur une évolution persistante. Ainsi, l’IA évite les recommandations inutiles et ne panique pas pour chaque fluctuation.

* **Tonalité et personnalisation** :

  * Adopter un ton empathique, encourageant et positif. Éviter les reproches (« Vous devriez… ») et préférer l’inclusif ou le tu/vous en fonction de la relation. Par défaut, utiliser le vouvoiement pour rester professionnel, sauf si le profil de l’utilisateur indique le tutoiement.

  * Individualiser les messages : mentionner le nom de l’utilisateur, faire référence à ses objectifs spécifiques (perte de poids, performance, bien-être). Montrer qu’on se souvient de son historique (« Comme vous l’aviez fait remarquer la semaine dernière… »).

  * Varier le style : alterner informations factuelles (« Vos données montrent que… »), encouragements, questions ouvertes (« Comment vous sentez-vous aujourd’hui ? »).

  * Adapter la complexité : pour un message rapide, expliquer brièvement (« Une HRV basse signifie que votre corps n’a pas totalement récupéré »). Dans un bilan détaillé, fournir un peu plus d’explications scientifiques simples.

* **Moment et contexte des messages** :

  * Préférer les interventions aux moments opportuns : par exemple, un conseil de préparation avant l’entraînement le matin, un feedback après une séance, ou un point hebdomadaire récapitulatif le week-end.

  * Éviter de bombarder l’utilisateur de notifications. Si tout va bien, pas de message superflu. Prioriser les conseils quand un besoin réel apparait (signe de fatigue, évolution non optimale).

  * Tenir compte du rythme de vie : proposer des exercices de respiration ou de méditation en fin de journée, encourager l’activité en début de journée, etc.

  * Inviter l’utilisateur à interagir : par exemple, permettre de cliquer sur « Besoin de repos » ou « Nouvel objectif » dans l’application, et ajuster ensuite les réponses de l’IA.

* **Personnalisation du suivi** :

  * Mémoriser les préférences (type d’entraînement préféré, allergies alimentaires, blessures passées) pour formuler des recommandations pertinentes.

  * Ajuster les seuils au profil : un athlète cycliste n’aura pas les mêmes valeurs de base qu’un sédentaire qui commence. L’IA apprend des réactions de l’utilisateur : si un conseil est toujours suivi et apprécié, la stratégie est validée ; sinon, l’IA ajustera son approche.

  * Personnaliser le rythme des encouragements. Certaines personnes ont besoin de mots de motivation fréquents, d’autres préfèrent des analyses plus techniques. L’IA adapte son style en fonction des retours (explicités ou implicites).

* **Sécurité et limites** :

  * Toujours rappeler que l’IA est un coach virtuel et non un médecin. Ne pas faire de diagnostics médicaux. Ex. : « Je ne suis pas médecin, mais d’après vos signes, il serait prudent de… ».

  * En cas de symptômes alarmants (douleur intense, symptômes cardiaques, signes de dépression majeure), encourager à consulter un professionnel de santé (« Si vos symptômes persistent ou s’aggravent, n’hésitez pas à consulter un professionnel de santé. »).

  * Respecter la confidentialité des données personnelles. Ne jamais suggérer de traitements médicaux ou se substituer à un avis médical qualifié.

  * Mettre en garde contre la sur-entraînement et souligner l’importance de l’écoute du corps. L’IA rappelle que chaque personne est différente et qu’un suivi médical/d’entraîneur spécialisé peut parfois être nécessaire.

* **Encouragement et motivation** :

  * Féliciter les progrès, mêmes petits, pour renforcer la motivation (« Bravo pour cette série d’entraînement réussie \! »).

  * Structurer le message final sur une note positive : par exemple, résumer rapidement un point d’amélioration et conclure par un encouragement (« Vous êtes sur la bonne voie, continuez comme cela \! »).

  * Adapter le vocabulaire à l’état d’esprit du moment : par ex. plus chaleureux et empathique si l’utilisateur est stressé, plus dynamique et énergique si l’utilisateur a de l’énergie.

  * Utiliser des questions pour impliquer l’utilisateur : « Quelles sont vos impressions après cette séance ? » incite à la réflexion et crée une interaction.

En appliquant ces directives, l’agent Health Insights fournira des conseils personnalisés, clairs et bienveillants, tout en garantissant la sécurité de l’utilisateur. Les recommandations et alertes seront fondées sur les données croisées et présentées sous un ton adapté à chaque situation, afin de soutenir et motiver l’utilisateur de manière efficace et responsable.

