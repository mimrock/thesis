# Történet

Ebben az időben a modern gépi tanulás még nehezen értelmezhető fogalom, mivel nem voltak számítógépek. Rengeteg, a gépi tanuláshoz szükséges matematikai eszközt viszont ebben az időben dolgoztak ki. Az egyik legfontosabb közülük talán Thomas Bayes, "An Essay towards solving a Problem in the Doctrine of Chances"(https://royalsocietypublishing.org/doi/10.1098/rstl.1763.0053) műve, ami - más fontos eredmények mellett feltételes valószínűségek kiszámításához használható - a ma Bayes-tételként ismert módszer alapjait is leírja. Érdekesség, hogy a mű két évvel Bayes halála után egy barátja, Richards Prince matematikus szerkesztésében jelenet meg.

Ezt 1814-ben Pierre-Simon Laplace finomította tovább a "Théorie Analytique des Probabilités" című könyvének második kötetében, ekkor született meg a Bayes-tétel, bár ezen a néven csak jóval később Henri Poincaré hivatkozott rá (https://mathshistory.st-andrews.ac.uk/Biographies/Laplace/).

A korai matematikai eredmények közül mindenképpen említésre méltó még Andew Markov 1913-as munkája (http://www.alpha60.de/research/markov/DavidLink_AnExampleOfStatistical_MarkovTrans_2007.pdf) amiben egy Puskin-művet, az Eugene Onegint próbálja meg statisztikai alapon értelmezni, és ezzel megalkotja a ma Markov-láncként ismert módszert.

1945 Decemberében bekapcsolták az első, gyakorlati felhasználásra szánt, elektronikusan programozható számítógépet, az ENIAC-ot(https://www.history.com/this-day-in-history/univac-computer-dedicated), amit ugyan nem használtak közvetlenül gépi tanulásra, de elindította a számítógépek fejlődését.

1950-ben Alan Turing az azóta rendkívül híressé vált, "COMPUTING MACHINERY AND INTELLIGENCE" című publikációjában bevezeti a "tanuló gépek" fogalmát(https://academic.oup.com/mind/article/LIX/236/433/986238), ami az előre betáplált szabályokkal ellentétben képes módosítani a működését az új információk hatására. Ugyanebben a publikációban azt javasolja, hogy ahelyett, hogy megpróbáljuk definiálni, hogy mit jelent egy gép számára a gondolkodás, egy imitációs játékra tesz javaslatot annak eldöntésére, hogy egy gép gondolkodik-e.

A játékban egy kikérdező, és (az eredeti verzióban) két játékos van. Az egyik játékos egy gép, a másik pedig egy ember. A kikérdező a játékosokkal csak írásban tud kommunikálni, és el kell döntenie, hogy melyik játékos ember, és melyik a gép. Ma ezt a játékot (és változatait) Turing-tesztnek nevezzük.

1951-ben  Marvin Lee Minsky megalkotja az első neurális hálót, bár a gyakorlat

1952 — Game of Checkers?


-----pm-----


## Naiv Bayes

A Naiv Bayes egy olyan, osztályozásra szolgáló algoritmuscsoport, ami a Bayes-tétel alapján működik. Naivnak azért nevezzük, mert feltételezi, hogy a megfigyelt tulajdonságok egymástól függetlenek. Bár a valóságban ez a feltételezés gyakran nem teljesül a tapasztalatok alapján a Naiv Bayes algoritmusok az egyszerűségük ellenére változatos feladattípusokban jól teljesítenek.

### Működési elv

Adottak a minták tulajdonságai, és az osztály, amibe soroljuk őket. A tanítás során a Naiv Bayes algoritmusok a megadott minták alapján a Bayes-i feltételes valószínűség elvét felhasználva meghatározzák, hogy az egyes tulajdonságok milyen mértékben korrelálnak a végső osztállyal. A predikció során pedig az új mintákat azokba az osztályokba sorolják, amelyek legnagyobb eséllyel járnak együtt az ismert tulajdonságokkal.

TODO: Képlet
TODO: Legalább 1 hivatkozás

------ gyak ----

## Felhasznált eszközök

### programozási nyelv

mik jöttek szóba. Végül a pythonra esett a választás, ilyen meg olyan ok miatt

### Python

A Python egy interpretált, magas szintű, általános célú programozási nyelv, amit Guido van Rossum fejlesztett ki 1990 körül. Erősen típusos, de a típusrendszere dinamikus. Mivel szemétgyűjtővel rendelkezik, ezért a memória lefoglalása és felszabadítása nem a programozó feladata. A Python fejlesztése során nagy hangsúlyt fektettek arra, hogy egyszerű, olvasható kód ok születhessenek. A python jelmondata "batteries included", azaz "elemmel a csomagban", ami arra utal, hogy a standard függvénykönyvtára szerteágazó, és harmadik féltől letöltött, külön könyvtárak nélkül is sokféle feladatot el tud végezni. Részben ezen tulajdonságai miatt a python az utóbbi időben rendkívül népszerűvé vált, különösen az adattudósok közt.[tiobe?]

A python támogatja az objektum-orientált programozást, bár az egyszerűségre való törekvése itt is megfigyelhető. Több, más objektum orientált nyelvekben megszokott eszköz hiányzik belőle. Így például nincs lehetőség a metódusok és tulajdonságok láthatóságának jelölésére. A jelenleg aktuális, hivatalos PEP 8 formázási javaslat ezért nem is privát, hanem nem-publikus tulajdonságokról és metódusokról beszél.  Ezek nevét aláhúzással szokás kezdeni, de ez nem jelent semmilyen, fordítió vagy futtatókörnyezet által nyújtott garanciát, csak a fejlesztők számára nyújt iránymutatást. Ugyanígy hiányoznak a Python-ból az interface-ek és absztrakt osztályok.

TODO: refs
e.g.: https://www.python.org/dev/peps/pep-0020/

### Scikit-learn

A scikit-learn egy nyílt forráskódú python függvénykönyvtár, ami a prediktív adatelemzésre szolgáló algoritmusok széles skáláját tesz elérhetővé. A projectet 2007-ben David Cournapeau hívta életre eredetileg a Google Summer of Code program keretében. Jelenleg, csak a githubon 205 ezer másik project használja.

### YAML

A YAML (YAML Ain't Markup Language) egy ember által jól olvasható adatszerializásra való fájlformátum. a JSON-nál alkalmasabb ember általi szerkesztésre, mert nem kell figyelni a zárójelekre, idézőjelekre, és vesszőkre. Részben emiatt az utóbbi évekre rendkívül népszerűvé vált, és legtöbb programnyelv már támogatja.

## Használat

### Előkészületek

Ahhoz, hogy a szoftvert használjuk, szükségünk van a megfelelő futtatási környezet telepítésére. Ennek a legfontosabb eleme egy python értelmező. Ez minden népszerűbb operációs rendszerhez ingyenesen letölthető, például Apple OSX, Microsoft Windows és Linux környezethez is[citation]. A szoftver használatát bemutató parancsok esetében feltételezett, hogy a python értelmező a `python` paranccsal indítható el (ellentétben például a Linuxon alapértelmezett python3 paranccsal). Szükség van még egy parancsértelmezőre is. Ez minden operációs rendszerben megtalálható, de a továbbiakban az OSX és Linux rendszerekben alapértelmezett bash[citation] értelmezőt vesszük alapul. Windows rendszer esetén a Microsoft által fejlesztett ingyenes WSL, esetleg a nyílt forráskódú, Git project részét képező Git Bash telepítésével szerezhetünk be bash-t. A szoftver természetesen más parancsértelmezőkkel (például PowerShellel, zsh-val, stb.) is kompatibilis, de a parancsok szintaktikája eltérhet.

A kényelmes munkához érdemes még beszerezni egy olyan szövegszerkesztőt, amivel a YAML fájlok szerkeszhetőek. Fontos, hogy a szövegszerkesztő ne tegyen extra, formázó karaktereket a szövegbe (text editor vs. word processor[citation]), és hasznos, ha képes YAML fájlok szintaktikai kiemelésére. Rengeteg ingyenes megoldás van, a teljesség igénye nélkül ilyen a Microsoft Windows platformon futó Notepad++ és VSCode, a bármely népszerű platformon futtatható Geany és Atom, és a Linuxon futtatható gedit. Én magam a JetBrains, IntelliJ Idea nevű IDE-jét használtam a YAML fájlok szerkesztéséhez.

### Telepítés

A program telepítése jelen esetben azon python csomagok letöltését jelenti, amiket a szoftver a működés során használ. Ezek a csomagok a szokásoknak megfelelően[citation] a `requirements.txt` nevű fájlban vannak felsorolva a kívánt verzióval együtt. A következő parancsok kiadása előtt lépjünk be abba a könyvtárban amiben a trainer.py, a szoftver fő fájla található. Bár a csomagokat lehetséges a globális környezetbe is telepíteni, javasolt létrehozni egy úgynevezett virtuális környezetet[internal reference] a következő paranccsal:

`$ python -m venv venv/`

Ez a parancs létrehoz a venv/ nevű könyvtárban egy virtuális környezetet. Linux vagy Apple OSX rendszer esetén aktiváljuk a 

`$ source venv/bin/activate`

paranccsal. Microsoft indows esetén az aktiválást a 

`$ ./venv/bin/activate`

paranccsal lehet elvégezni.

Az aktiválás után a bash promptja zárójelben jelzi, hogy a virtuális környezet aktiválva van, pl.:

`(venv) denes@denes-desktop$`

Most már minden python csomag, amit ezután telepítünk a virtuális környezetbe kerül a globális környezet helyett. A szoftver futtatásához szükséges csomagokat a python részeként terjesztett[cit] pip csomagkezelő segítségével lehet feltelepíteni.

`$ pip install -r requirements.txt`

Ha a telepítés hiba nélkül lezajlott, akkor ezután megkezdhetjük a program használatát.

### Előzetes jelentés

Feltéve, hogy az adataink a data/data.csv-ben vannak, generáljunk egy előzetes jelentést a következő paranccsal:

`$ python trainer.py --generate-report data/data.csv --output report.html`

Ez a parancs egy html fájlt fog generálni, ami a data.csv-ben lévő adatokról szolgáltat információkat[internal ref]. Ezen információk birtokában könnyebb megtervezni az adatok feldolgozását.

### Egy futási terv készítése

Ahhoz, hogy az egyszerű korreláción túlmutató összefüggéseket találjunk az adatainkban, egy futtatási terv létrehozására van szükség. Ez a futtatási terv YAML formátumú. Egy, a példa kedvéért az átlagosnál sokkal egyszerűbb futtatási terv a következőképpen néz ki:

```
data_file: 'data/data.csv'
test_ratio: 0.2
mapping:
   id:
      use: ignore
   x_a:
      encode: "one_hot"
   x_b:
      encode: "default"    
   y:
      use: "target"  
```

Láthatjuk, hogy a legfölső szinten három beállítást tudunk megadni - mind a három kötelező.

1. data_file: Az adatokat tartalmazó, fejléces csv fájl elérési útja. Lehet abszolút, vagy relatív út is. Utóbbi esetben a program futtatásának helyéhez képest oldódik fel.

2. test_ratio: Azt adja meg, hogy az adatok mekkora részét különítse el a szoftver a generált modellek validálására. A validáláshoz futtatásonként egyszer, véletlenszerűen választódnak ki a sorok.

3. mapping: Ez egy komplex mező, ahol a modellek megalkotásához szükséges mezők kezelését lehet megadni. A mapping mező további komplex mezőket tartalmaz. A felsorolt mezők nevei meg kell hogy egyezzenek az adatokat tartalmazó csv fájlban megadott oszlopnevekkel. Az oszlopokat jelképező mezőknek két további mezője van a use és encode. A szoftver csak azokat a mezőket veszi figyelembe, amiket expliciten megadunk, de lehetőség van expliciten kiírni, ha egy mezőt - a példában az id mezőt - ignorálni szeretnénk.
    1. use: Azt lehet megadni vele, hogy mi legyen a szerepe az adott mezőnek a modellek felépítése során. Három lehetséges értéke van: ignore, target és feature. Ignore esetén a szoftver nem foglalkozik a mezővel egyáltalán. Ez az alapértelmezett érték. feature esetén a mező a modell egyik tulajdonsága(feature) lesz. target esetén a mező lesz a célváltozó. Figyeljünk, hogy pontosan egy mezőt jelöljünk meg target-ként.
    2. encode: A lehetséges értékei: original és one_hot. Csak feature-ként megadott mezők esetén veszi figyelembe a rendszer. original esetén a mező tartalma változtatás nélkül kerül a modellbe. one_hot esetén a mezőt a modellbe illesztés előtt one hot[iref] kódolás segítségével bináris kategorikus változókká alakítja a rendszer.
    
### Futtatási terv végrehajtása

Ha elkészítettük a futtatási tervünket, mentsük el, a példa kedvéért plan.yaml néven a trainer.py mellé. Ezután adjuk ki a következő parancsot

`$ python trainer.py --execute_plan --planfile=plan.yaml`

A program az eredményét a standard kimenetre fogja kiírni, a futás közben keletkező információkat pedig a standard hibakimenetre. Alapértelmezetten mindkettő a képernyőn jelenik meg. 

### Adatok

Az adatokat fejléces csv formátumban kell a szoftver rendelkezésére bocsátani utf-8 karakterkódolásban. Bár nem feltétel, de érdemes ügyelni rá, hogy a fejlécben az oszlopok nevei csupa kisbetűvel legyenek elnevezve, ugyanis a szofver megkülönbözteti a kis és nagy betűket. Fontos, hogy a csv fájl olyan helyre kerüljön, ahol a program eléri azt, nem akadályozza jogosultsági, vagy egyéb probléma. A fájlnak nincs mérletkorláta, elvileg bármennyi adatot tartalmazhat, de tekintettel kell lenni arra, hogy a különböző műveletek, különösen egyes tanulóalgoritmusok illesztési szakasza nem lineáris futási idejű (például egy SVN polynomiális kernellel négyzetes időben képes csak végrehajtani az illesztést), ezért attól függően, hogy milyen műveleketeket szeretnénk végezni az adatokkal, egy nagyobb adatfájl feldolgozása irreálisan sok időt igényelhet.

# szerkezet

1. bevezetés (célkitűzések, indoklás, probléma ismeretése röviden)
2. tárgyalás ()
    x. némi elmélet? 1, 2, vagy 3
    1. ml történet
    2. ml a napjainkban (pl. példák, belfry?)
    3. ml tipikus munkafolyamat, mi az a pipeline (validation, evaluation)
    4. pipeline követelmények (általánosságok)
    5. tervezés, fejlesztés lépései (cli-ból használható, felhasznált technikák, kiválasztásuk indoklása, mit nem)
       - alternatívák felvetése a technológia fejezetekben (pl. yaml fejezet alatt json, toml említése, a cím pedig általános) 
    6. manual
    7. esettanulmány (kaggle adatok bemutása, miért hasznos, eredmények)
    8. összefoglalás, eredmények, dolgozaton túlmutató további teendők
