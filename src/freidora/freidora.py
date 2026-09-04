""" 
    Libreria RISC para uso de pandas de forma simplificada.

    Algunas decisiones de diseño:
      * Las operaciones se hacen por omisión inplace, de forma opuesta a pandas
      * Todas las operaciones devuelven el dataset, independientemente de si son inplace o no
      * Elimina todos los campos y índices anidados que genera pandas. De forma que las 
        tablas tras las operaciones siguen siendo planas (campos e indice simple)
      * Trata al íncide como si fuera un campo más

    Ademas la libreria permite rejecutar y auditar secuencias de transformaciones realizadas previamente
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import inspect
import warnings
import copy
import dis
import types
import dill
import zipfile as zf
import graphviz

from abc import ABC
import os

########## FUNCIONES DE AUDITORIA ####################
#

def get_used_globals(code):
    """Recursively find all global names referenced in a code object."""
    names = set()
    for instr in dis.get_instructions(code):
        if instr.opname == "LOAD_GLOBAL":
            names.add(instr.argval)
    # Recurse into nested functions/comprehensions
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            names |= get_used_globals(const)
    return names

def make_cell(val):
    """Create a new closure cell containing val."""
    def inner():
        return val
    return inner.__closure__[0]
    
def copy_func(f, _visited=None):
    def _try_copy(v):
        try:
            return copy.copy(v)
        except Exception:
            return v
    if _visited is None:
        _visited = {}
    if id(f) in _visited:
        return _visited[id(f)]

    used = get_used_globals(f.__code__)
    globals_snapshot = {}
    _visited[id(f)] = None

    for k, v in f.__globals__.items():
        if k not in used:
            continue
        if isinstance(v, types.FunctionType):
            v = copy_func(v, _visited)
        else:
            v = _try_copy(v)
        globals_snapshot[k] = v

    new_closure = None
    if f.__closure__:
        def copy_cell_value(v):
            if isinstance(v, types.FunctionType):
                return copy_func(v, _visited)  # recursively freeze functions in closures
            return _try_copy(v)

        new_closure = tuple(make_cell(copy_cell_value(c.cell_contents)) for c in f.__closure__)

    new_f = types.FunctionType(
        f.__code__, globals_snapshot, f.__name__, f.__defaults__, new_closure
    )
    new_f.__kwdefaults__ = f.__kwdefaults__
    _visited[id(f)] = new_f
    return new_f

class IteradorRefrito():
    
    def __init__(self,d,r):
        self.d = d
        self.r = r
        
    def __iter__(self):
        self.previos = [(self.d, self.r.pasos[self.d][-1], len(self.r.pasos[self.d])-1,"")]
        self.i = len(self.r.pasos[self.d])
        return self

    def __next__(self):
        if not self.previos:
            raise StopIteration
        datos, paso, i, dependencias = self.previos.pop()
        profundidad = len(dependencias)
        if i == 0 and not paso.esHoja():
            for d in paso.datos_in[::-1]:
                dd = "0" if d==paso.datos_in[-1] else "1"
                self.previos.append((d, self.r.pasos[d][-1], len(self.r.pasos[d])-1, dependencias+dd))
        elif i > 0:
            self.previos.append((datos, self.r.pasos[datos][i-1], i-1, dependencias))
        txt = "    " + dependencias.replace("0","    ").replace("1", "|   ")
        if self.r.pasos[datos][-1] == paso:
            print(txt[:-4]+"|")
            txt = txt[:-4] + "----" 
        print(txt, end="")
        return paso#self.r.pasos[self.d][self.i]
        

class Refrito(ABC):

    CONTAR = True
    recalentador = None
    
    def __init__(self):
        self.result = None
        self.pasos = {}
        
    @staticmethod
    def getRecalentador():
        if Refrito.recalentador is None:
            Refrito.recalentador = Refrito()
        return Refrito.recalentador

    def iterar(self,datos):
        return IteradorRefrito(datos,self)
        
    def guardar(self, datos, nfichero, limpio=True, guarda_datos=True, en_zip=True, 
                otros_ficheros=[]):
        """ Guarda la secuencia de obtención de 'datos' al fichero 'nfichero'.pkl.
            
            - limpio: No sé qué hace esto, no se usa 
            - guarda_datos: Se guarda los datos de origen además de las operaciones si True
                      Si False no (útil si los ficheros de datos de origen no cambian). 
            - en_zip: Los datos seguardan en un zip con igual nombre base. Si no se guardan 
                      en la propia secuancia
            - otros_ficheros: a guardar en el zip, por ejemplo el código fuente
        """
        
        self.recalentar(datos, forzar_ejecucion = True, guardar_datos_cruce = False,
                               guardar_datos_hoja = guarda_datos and not en_zip, 
                               en_seco = True)
        
        with open(f'{nfichero}.pkl', 'wb') as fich:
            dill.dump(Refrito.recalentador, fich)
            
        if en_zip or otros_ficheros:
            ficheros = set(otros_ficheros)
            if en_zip:
                for n in Refrito.getRecalentador().iterar(datos):
                    fichero = n.esNodoLecturaDatos()
                    if fichero:
                        ficheros.add(fichero)
            with zf.ZipFile(f'{nfichero}.zip', mode="w", compression=zf.ZIP_DEFLATED, allowZip64=True,compresslevel=9) as z:#ZIP_STORED
                for fichero in ficheros:
                    z.write(fichero)
       
    def cargar(self, nfichero):
        with open(f'{nfichero}.pkl', 'rb') as fich:
            Refrito.recalentador = dill.load(fich)
        
    def recalentar(self, datos, forzar_ejecucion=False, guardar_datos_cruce=True, guardar_datos_hoja=False, en_seco=False):
        return self.pasos[datos][-1].run(forzar_ejecucion, guardar_datos_cruce, guardar_datos_hoja, en_seco)

    def mostrar(self, datos):
        if Refrito.CONTAR:
            self.recalentar(datos, en_seco=True)
        self._mostrar(datos, "0", -1)
        Refrito.CONTAR = False
        
    def _mostrar(self, datos, indent="0", ipaso=-1):
        """ Este método vuelve a ejecutar la secuencia de funciones 
            de freidora ejecuadas sobre un data frame """
        
        txt_indent = ""
        for c in indent:
            txt_indent += "|   " if c=="1" else "    "
        
        if datos not in self.pasos:
            return
            
        prev = txt_indent[-4:]
        txt_indent = txt_indent[:-4]
        print(txt_indent + "|")
        
        if ipaso<0: 
            ipaso = len(self.pasos[datos])
            
        for i, paso in enumerate(self.pasos[datos][::-1]):
            if len(self.pasos[datos])-i > ipaso:      add = "|   +"
            else:
                if len(self.pasos[datos])-i == ipaso: add = "----+" 
                else:                                     add = prev + "+"
                    
                if Refrito.CONTAR: paso.pasos.append("*")
            
            print(txt_indent+add+f" {paso}") 

        # Recursión
        paso = self.pasos[datos][0]
        for i, (d,ipaso) in enumerate(zip(paso.datos_in, paso.nums)):
            if d != datos:
                self._mostrar(d,indent+("1" if i+1 < len(paso.nums) else "0"), ipaso)
        
    def build_graphviz(self, datos_raiz):
        """
        Construye un Digraph de graphviz a partir del árbol de pasos que
        desemboca en el dataframe identificado por datos_raiz.
        """
        def _label_paso(paso: PasoRecalentado) -> str:
            """Genera la etiqueta de texto de un nodo."""
            func_name = paso.funcion.__name__
                
            if paso.esNodoLecturaDatos():
                return f"carga_tabla\n{paso.nombre_fichero}"
                        
            if not paso.params:
                return func_name
        
            parts = []
            for k, v in paso.params.items():
                if v is None:
                    continue
                if callable(v):
                    parts.append(f"{k}={v.__name__}")
                else:
                    s = str(v)
                    parts.append(f"{k}={s[:25]}{'…' if len(s) > 25 else ''}")
        
            return func_name + "\n" + "\n".join(parts) if parts else func_name
        
        
        def _node_style(paso: PasoRecalentado) -> dict:
            """Devuelve atributos de estilo graphviz según el tipo de nodo."""
            if paso.esNodoLecturaDatos():
                return {"shape": "cylinder", "style": "filled", "fillcolor": "#AED6F1"}
            if paso.nodo_no_trazable:
                return {"shape": "box", "style": "filled,dashed", "fillcolor": "#FAD7A0"}
            if paso.tieneDatos():
                return {"shape": "box", "style": "filled", "fillcolor": "#F5A623"}
            return {"shape": "box", "style": "filled", "fillcolor": "#F0F0F0"}

        dot = graphviz.Digraph(graph_attr={"rankdir": "TB", "splines": "ortho"})
        visited: set[str] = set()
    
        def add_paso(paso: PasoRecalentado):
            nid = str(paso.identificador)
            if nid in visited:
                return
            visited.add(nid)
    
            label = _label_paso(paso)
            style = _node_style(paso)
            dot.node(nid, label=label, **style)
    
            for d_in, num in zip(paso.datos_in, paso.nums):
                previo = self.getPrevio(paso, d_in, num)
                if previo is None: 
                    continue
                add_paso(previo)
                dot.edge(str(previo.identificador), nid)
    
        if datos_raiz in self.pasos and self.pasos[datos_raiz]:
            add_paso(self.pasos[datos_raiz][-1])
    
        return dot
    def añadirPaso(self, funcion, datos_in, datos_out, params=None, lista_de_entrada=False, otros={}):
        id_datos_out = id(datos_out)
        if id_datos_out not in self.pasos:
            self.pasos[id_datos_out] = []
        if type(datos_in)!=tuple:
            datos_in = (datos_in,)
        datos_in = tuple([id(di) for di in datos_in])
        nums = [len(self.pasos[d_in]) if d_in in self.pasos else 0 for d_in in datos_in]
        paso = PasoRecalentado(self.recalentador, funcion, datos_in, datos_out, params, 
                               nums,# if id(datos_in[0])!=id(datos_out) else [],
                               lista_de_entrada, otros)
        self.pasos[id_datos_out].append(paso)

    def getPrevio(self, paso, d_in, num):
        #print(f"buscando previo {d_in}{paso.funcion.__name__}")
        if paso.datos_out==d_in:
            i = self.pasos[paso.datos_out].index(paso)
            return self.pasos[paso.datos_out][i-1] if i > 0 else None
            
        return self.pasos[d_in][num-1] if d_in in self.pasos else None
        
    def esUltimoPasoSecuencia(self, paso):
        return self.pasos[paso.datos_out][-1]==paso and paso.datos_out==paso.datos_in[0]
        

class PasoRecalentado:
    def __init__(self, recalentador, funcion, datos_in, datos_out, params=None, 
                 nums=[], lista_de_entrada=False, otros={}):
        self.recalentador = recalentador
        self.identificador = id(self)
        self.funcion   = funcion
        if type(datos_in)!=tuple:
            datos_in = (datos_in,)
        self.datos_in  = datos_in
        self.datos_out = id(datos_out)
        self.params    = params
        self.nums = nums
        self.pasos = []
        self.resultado = None
        self.lista_de_entrada = lista_de_entrada
        self.otros = otros
        if self.esNodoLecturaDatos():
            self.nombre_fichero = carga_tabla(self.params["nombre"], devuelve_solo_nombre_fichero=True)
        # Indica un nodo que tiene datos que no han sido capturados por freidora, lo que hacer que
        # se pierda la trazabilidad, en estos casos se guardan los datos
        self.nodo_hoja = len(nums)==1 and nums[0]==0
        self.nodo_no_trazable = self.nodo_hoja and not self.esNodoLecturaDatos()
        if self.nodo_no_trazable:
            self.salida_no_trazable = datos_out.copy()
            #display(self.salida_no_trazable)
            print("No trazable", self)

    def __eq__(self, otro):
        try:
            return self.identificador == otro.identificador
        except:
            pass
        return False
        
    def esHoja(self):
        return self.nodo_hoja
        
    def tieneDatos(self):
        return self.resultado is not None
        
    def esNodoLecturaDatos(self):
        """ Devuelve el nombre de fichero si es de lectura y si no lo es devuelve None (esto avalua a true/false)"""
        return self.otros["nombre_fichero_abierto"] if self.funcion.__name__ == 'carga_tabla' else None
        
    def run(self, forzar_ejecucion=False, guardar_datos_cruce=True, guardar_datos_hoja=False, en_seco=False):
        if en_seco: self.pasos = []
            
        if self.nodo_no_trazable:
            print(f"Running {self} NO TRAZABLE!")
            return self.salida_no_trazable.copy()
        if self.resultado is not None and not forzar_ejecucion:
            return self.resultado
            
        ins = []
        for d_in, num in zip(self.datos_in, self.nums):
            previo = self.recalentador.getPrevio(self, d_in, num)
            if previo:
                ins.append(previo.run(forzar_ejecucion, guardar_datos_cruce, guardar_datos_hoja, en_seco))

        if en_seco and not (guardar_datos_hoja and self.nodo_hoja and self.resultado is None):
            print(f"Running en seco {self}")
            res = None
        elif self.lista_de_entrada:
            res = self.funcion(ins, **self.params, track_changes=False)
        else:
            res = self.funcion(*ins, **self.params, track_changes=False)

        self.resultado = res if guardar_datos_hoja and self.nodo_hoja else None
        if self.resultado is None:
            #Solo se guarda si es un paso cruce, el final de una secuencia de modificaciones inplace
            self.resultado = res if guardar_datos_cruce and self.recalentador.esUltimoPasoSecuencia(self) else None
            
        return res

    def __str__(self):
        a, c = ("","") if self.resultado is None else ("[[[","]]]")
        d = {k:(v.__name__ if k=='fun' else str(v)) for k,v in self.params.items() if v is not None}
        return f"{a}({self.nums}-{"".join(self.pasos)}): {self.funcion.__name__}({",".join([k+"="+v for k,v in d.items()]) }){c}"


########## FUNCIONES GENERICAS de PROCESADO ####################
# Estas funciones trabajan con dataframes de pandas
#

def s2f(fecha):
    """ Funcion de conversion de fecha
    """
    if pd.isnull(fecha):
        return np.nan#fecha
    if type(fecha) is datetime:
        return fecha
        
    try:
        #return pd.to_datetime(fecha,dayfirst=True).date() # No lee 1/1/3000
        s = '/' if '/' in fecha else '-'
        format = "%Y-%m-%d" if fecha[4]=='-' else "%d-%m-%Y"
        format = format.replace('-',s)
        r = datetime.strptime(fecha,format).date()
    except:
        try:
            format = "%Y-%m-%d %H:%M:%S" if fecha[4]=='-' else "%d-%m-%Y %H:%M:%S"
            format = format.replace('-',s)
            r = datetime.strptime(fecha,format).date()
        except:
            print('Error en conversion a fecha de', fecha, type(fecha))
            return np.nan
        
    return r

def carga_tabla(nombre, campos=None, devuelve_solo_campos=False, devuelve_solo_nombre_fichero=False, 
                track_changes=True):
    """
        Hay que hacer esta función más genérica y no tan orientada a abrir csv, algo para abrir
        usando cualquier param de pandas (por ejemplo para abrír hojas específicas de excel que 
        ahora no está).
        
        Funcion para abrir tablas csv, etc con distintos metodos.
        Busca el método adecuado de apertura ya que cada tabla está en un formato. 
        Prueba las siguientes posibilidades: 
            1.- csv sin configurarcion adicional
            2.- csv separado por ; y con codificacion windows cp1252
            3.- csv separado por ; y con codificacion utf8
            4.- como 1.- pero con engine=python que estima el separador
            5.- excel o ods

        El orden depende de si es csv y de una estimación del separador mirando la primera 
        línea del fichero (si es csv)
        
        Parámetros:
        
        * nombre: nombre de la tabla a abrir. No se necesita poner la extensión: busca csv, xlsx, ods. 
                  Si se quiere abrir ods se debe poner la extensión
        * campos: Si se especifica campos, solo carga los campos (columnas) indicadas
        * devuelve_solo_campos: Se es True entonces devuelve la lista de campos disponibles (solo válido para csv)
        * devuelve_solo_nombre_fichero: Devuelve el nombre de fichero que ha localizado para abrir
        
        Devuelve un dataframe de pandas
    """
    import glob

    params = {"nombre":nombre, "campos":campos}
    
    #f_gen  = lambda f:pd.read_csv(f,sep=None,engine='python',skipinitialspace=True)
    f_gen  = lambda f:pd.read_csv(f,skipinitialspace=True,usecols=campos)
    f_esp  = lambda f:pd.read_csv(f,sep=";",encoding='cp1252',thousands='.', \
                                  decimal=',',skipinitialspace=True,usecols=campos)
    f_esp2 = lambda f:pd.read_csv(f,sep=";",encoding='utf-8', thousands='.', \
                                  decimal=',',skipinitialspace=True,usecols=campos)
    f_lat  = lambda f:pd.read_csv(f,sep=None,engine='python',encoding='cp1252',\
                                  skipinitialspace=True,usecols=campos)
    
    funciones = []
    if len(glob.glob(nombre))==1:
        pass # Nos dan el nombre completo del fichero con extensión
    elif len(glob.glob(nombre+".csv"))==1: 
        nombre += '.csv' # Existe el fichero con csv
    elif len(glob.glob(nombre+".xlsx"))==1: 
        nombre += '.xlsx' # Existe fichero xlsx, 
    elif len(glob.glob(nombre+".ods"))==1: 
        nombre += '.ods' # Si no probamos con ods
    else:
        assert False, f"No existe ningún fichero de datos con nombre {nombre}"

    if nombre[-3:]=='csv':
        # Dependiendo de separadores de la primera fila ponermos un orden u otro para analizar
        with open(nombre, errors='ignore') as f:
            linea = f.readline()
            if linea.count(",")>linea.count(";"):
                funciones.extend([f_gen, f_esp, f_esp2, f_lat])
            else:
                funciones.extend([f_esp, f_esp2, f_gen, f_lat])
    else:
        funciones.append(lambda f:pd.read_excel(f,usecols=campos))

    if devuelve_solo_campos:
        # solo para csv
        with open(nombre,'r') as f:
            for line in f:
                sep = "," if line.count(',')>line.count(';') else ";"
                return line.strip().replace('"',"").split(sep)
    elif devuelve_solo_nombre_fichero:
        return nombre

    print("Abriendo: " + nombre)
    df = None
    if df is not None:
        # Esto era para ahorrar memoria, es difícil saber si el dataframe ha cambiado así 
        # que demomento no se usa
        print("Cargando " + nombre + " de memoria")
        return df
    else:
        pass

    for fun in funciones: # Se recorren las funciones de apertura hasta que alguna funciona
        try:
            df = fun(nombre)
            break
        except Exception as e:
            print('Nop con '+ nombre + " " + str(fun),e)
            
    if df is None:
        assert False, f"No se ha podido abrir el fichero {nombre}"
        return None
        
    from pandas.api.types import is_datetime64_any_dtype as is_datetime

    convert_mode = "B"
    for col_name in df.columns: # PAsa campos de fechas a tipo fecha
        if col_name.startswith('FC_') or col_name.startswith('FECHA_') or col_name.startswith('F__'):
            # Convertimos los campos de fechas, al leer de csv y con campos de fechas con 
            # distintos formatos no se cargan bien. 
            #if pd.core.dtypes.common.is_datetime_or_timedelta_dtype(df[col_name]):
            if convert_mode == "B":
                print('Convirtiendo '+col_name+' a fecha desde ',df[col_name].dtype)
                # El erros='coerce' no captura si el año es 0
                kfun = lambda r: "1900" + r[col_name][4:] if type(r[col_name]) is str and r[col_name][:4]=='0000' else r[col_name]
                map_columna(df, kfun, col_name, track_changes=False)
                df[col_name] = pd.to_datetime(df[col_name], format='%Y-%m-%d').dt.date
            else:
                if is_datetime(df[col_name]):
                    df[col_name] = pd.to_datetime(df[col_name])
                    df[col_name] = df[col_name].apply(lambda r:r.date())
                elif not df[col_name].dtype is datetime:
                    print('Convirtiendo '+col_name+' a fecha desde ',df[col_name].dtype)
                    df[col_name] = df[col_name].apply(s2f)
                else:
                    print(col_name, df[col_name].dtype,datetime, df[col_name].dtype is datetime)
        elif convert_mode == "A" and col_name.startswith('ID_'):
            print('Convirtiendo '+col_name+' a int desde ',df[col_name].dtype)
            # Convertimos campos id a int, que pandas los carga como float
            def to_int(r, campo):
                value = r[campo]
                if type(value)==int: return value
                try:
                    float_value = float(r[campo])
                    int_value = int(r[campo])
                except:
                    return value
                return int_value if float_value==int_value else value
                    
            map_columna(df, lambda r: to_int(r,col_name), col_name, track_changes=False)

    if convert_mode == "B":
        print("Convirtiendo ids de golpe")
        df = df.astype({col: "int32" for col in  df.columns if col_name.startswith('ID_')}, errors='ignore')
        print("Hecho")
        
        
    #df = df if not campos else select(df,campos,inplace=True)
    
    if track_changes:
        Refrito.getRecalentador().añadirPaso(carga_tabla, None, df, params, otros={"nombre_fichero_abierto":nombre})

    return df

def carga_optimizada(funcion_generacion_tablas, nombre_tablas_preprocesadas, tablas_involucradas):
    """
        recibe una función sin params que devuelve un(os) dataframe(s) que se guardará en 
        disco con el/los nombre_tablas_procesadas. Esta función declara que va a usar 
        las tablas_involucradas.

        En las sucesivas veces que se llame a la función, se comprobará si la fecha de 
        nombre_tablas_generadas es posterior a las tablas involucradas, si es posterior 
        se lee(n) la/las tabla(s) de fichero y se devuelve(n). Si no, se vuelve a llamar 
        a la función y se actualiza(n) la/las tabla_procesada(s) en disco
    """
    if type(nombre_tablas_preprocesadas) is str:
        nombre_tablas_preprocesadas = [nombre_tablas_preprocesadas]

    # Compruebo solo la primera, las otras se supone que se han generado a la vez
    if os.path.isfile(nombre_tablas_preprocesadas[0]) and \
        all([os.path.getmtime(nombre_tablas_preprocesadas[0])>os.path.getmtime(t) for t in tablas_involucradas]):
        return carga_tabla(nombre_tablas_preprocesadas[0]) if len(nombre_tablas_preprocesadas)==1 \
                                        else tuple([carga_tabla(tn) for tn in nombre_tablas_preprocesadas])
        
    tt = funcion_generacion_tablas()
    if type(tt) is not tuple:
        tt = (tt,)
    
    assert len(tt)==len(nombre_tablas_preprocesadas), "Las tablas y los dataframes que devuelve la función no coinciden"

    for t,nombre_tabla_preprocesada in zip(tt, nombre_tablas_preprocesadas):
        t.to_csv(nombre_tabla_preprocesada, index=False)
    
    return tt[0] if len(tt)==1 else tt

def filter(data, fun, inplace=True, track_changes=True):
    """ Interfaz en ingles """
    return filtra(data, fun, inplace, track_changes)
    
def filtra(data, fun, inplace=True, track_changes=True):
    """ Se aplica la funcion fun a todos los registros de data
        y se eliminan lo que la funcion fun devuelve True. Si 
        inplace=True se hace sobre el mismo dataframe, si no se 
        crea uno nuevo
        
        Si la entrada es un DataFrame se aplica drop
        
        Se la entrada es un dict, se recorren los elementos
    """
    
    data_in = data
    params = {"inplace":inplace,"fun":copy_func(fun)}
    
    name = data.index.name
    
    data_origen = data
    data_origen.reset_index(inplace=True)
    
    if inplace:
        data.drop(data[data.apply(fun, axis=1)].index, inplace=True)
    else:
        data = data.drop(data[data.apply(fun, axis=1)].index, inplace=False)
        data.set_index(name if name else 'index', inplace=True)
        data.index.name = name

    data_origen.set_index(name if name else 'index', inplace=True)
    data_origen.index.name = name

    if track_changes:
        Refrito.getRecalentador().añadirPaso(filtra, data_in, data, params)
    
    return data

def rename_columns(data, col_names, col_new_names, track_changes=True):
    """Interfaz en ingles"""
    return renombra_columnas(data, col_names, col_new_names, track_changes)
    
def renombra_columnas(data, cols_origen, cols_nuevos, track_changes=True):
    """ 
       Renombra la lista de columnas cols_origen por los nuevos valores cols_nuevos
    """

    params = {"cols_origen":cols_origen,"cols_nuevos":cols_nuevos}
    
    columns = list(data.columns)
    
    for o,f in zip(cols_origen, cols_nuevos):
        if o != f:
            columns[columns.index(o)] = f
        
    data.columns = columns

    if track_changes:
        Refrito.getRecalentador().añadirPaso(renombra_columnas, data, data, params)
    
    return data
    
def join(data1, data2, how='left',fields_1 = None, fields_2 = None, \
           verbose = True, only_extend=False, track_changes=True):
    """Interfaz en ingles"""
    return ajunta(data1, data2, how, fields_1, fields_2, verbose, only_extend, track_changes)
    
def ajunta(data1, data2, how='left',campos1 = None, campos2 = None, \
           verbose = True, only_extend=False, track_changes=True):
    """
       Función para hacer el equivalente a un JOIN de SQL
       
       Si solo pasas las dos tablas, hace un left join de ambas tablas 
       por los nombres de columnas iguales.

       Si se pone only_extend a true comprueba que el join solo tenga 1 registro
       en right para cada registro en left (es decir que sea m a 1)
       
       Puedes cambiar el comportamiento con los params
    """
    
    data_in = data1, data2
    params = {"campos1":campos1, "how":how,"campos2":campos2}
    
    if campos1 is None and campos2 is None:
        s1 = set(data1.columns)
        if type(data1.index.name) == str:
            s1.add(data1.index.name)
        s2 = set(data2.columns)
        if type(data2.index.name) == str:
            s2.add(data2.index.name)
        campos1 = campos2 = list(s1.intersection(s2))
        if verbose:
            print("Fusionando por campos:", campos1)
            
    elif campos2 is None:
        if type(campos1) is str:
            campos1 = [campos1]
        campos2 = campos1
        
    elif campos1 is None:
        if type(campos2) is str:
            campos2 = [campos2]
        campos1 = campos2
    
    # the reset_index and set-index is to keep the index after the merge
    # also to use the index as join field as if any other field
    data2 = data2 if data2.index.name not in campos2 else data2.reset_index() 
    result = data1.reset_index().merge(data2, how=how, left_on=campos1, \
                   right_on=campos2, indicator = True, validate="m:1" if only_extend else None)
    
    # To avoid lossing the index 
    result.set_index('index' if data1.index.name is None else data1.index.names, inplace=True)
    result.index.name = None if result.index.name == 'index' else result.index.name
    
    if verbose:
        rr = np.array(result['_merge'])
        
        print('no de datos: T1',len(data1),'+ T2', len(data2),'= R',len(result))
        print('provienen de left_only/both/right_only:',end=' ')
        print(np.sum(rr=='left_only'),"/",np.sum(rr=='both'),"/",np.sum(rr=='right_only'),sep="")
        
    result.drop(columns=['_merge'], inplace=True)
    
    if track_changes:
         Refrito.getRecalentador().añadirPaso(ajunta, data_in, result, params)
                               
    return result

def select(data, fields, inplace=True, en_orden=False, track_changes=True):
    """Interfaz en ingles"""
    return selecciona(data, fields, inplace, en_orden, track_changes)
    
def selecciona(data, campos, inplace=True, en_orden=False, track_changes=True):
    """ 
        Función para hacer el equivalente a un SELECT de SQL

        Selecciona los campos del dataframe data. 

        Si en_orden es True los devuelve en el orden especificado. 
        No se puede hacer en_orden y inplace
    """
    assert not (inplace and en_orden), "No se puede hacer inplace y en orden"
    
    data_in = data
    params = {"campos":campos, "inplace":inplace, "en_orden":en_orden}
    
    for campo in campos:
        if campo not in data.columns:
            warnings.warn("El campo " + campo + " no está en la tabla!!")
            
    if not en_orden:
        if inplace:
            data.drop(set(data.columns) - set(campos),axis=1, inplace=inplace)
        else:
            data_origen = data
            data = data.drop(set(data.columns) - set(campos),axis=1, inplace=inplace)
            
        if track_changes:
            Refrito.getRecalentador().añadirPaso(selecciona, data_in, data,{"campos":campos, "inplace":inplace, "en_orden":en_orden})
            
        return data

    data = data[campos].copy()
    
    if track_changes:
        Refrito.getRecalentador().añadirPaso(selecciona, data_in, data, params)
            
    return data

def map_columna(data, fun, column, method=1, track_changes=True):
    """
        Crea campos calculados
        
        Recorre data y crea, o actualiza si ya existe, el valor de la columna column.
        Si la columna column es una lista de columnas, entonces fun debe devolver 
        una lista de valores de igual longitud

        EL parámetro method indica posibles formas de hacerlo, estoy probando cuál es 
        más rápido. Quería probar método vectorial
        
    """
    data_in = data
    params = {"column":column, "fun":copy_func(fun), "method":method}
    
    if len(data)==0: 
        data[column] = None
        return data
    if method == 1:
        data[column] = data.apply(fun, axis=1, result_type='expand')
    elif method == 2:
        #assert False, "Este método no furula, o sí pero es lento"
        #No es más rápido funciona
        #data[column] = [fun(row) for i,row in data.iterrows()]
        data[column] = fun(data)

    if type(column) is not list:
        column = [column]
            
    if track_changes:
        Refrito.getRecalentador().añadirPaso(map_columna, data_in, data, params)
        
    return data

def map_bloque(data, fun, key_columns, column, method=1, track_changes=True):
    """         
        Es un map pero por bloques, es decir le va enviando a la función 
        los bloques de registros con una combinación de claves (key_columns) 
        únicas. La función debe devolver un listado con los valores nuevos 
        para column de igual longitud al número de registros recibidos

        Esta función es un poco lenta, hay varios metodos en pruebas
        
    """
    #assert type(key_columns)!=list
    #assert type(key_columns)==str
    data_in = data
    params = {"key_columns":key_columns, "column":column, "fun":copy_func(fun),"method":method}
    
    if method==1:
        if type(key_columns)==list and len(key_columns)==1:
            key_columns = key_columns[0]
            
        if type(key_columns)==list:
            campo_clave = "_".join(key_columns)
            map_columna(data, lambda r:"".join([str(r[f]) for f in key_columns]), campo_clave)
        else:
            campo_clave = key_columns
            
        if type(campo_clave)==str:
            keys = set(data[campo_clave])
            for key in keys:
                bloque = data.loc[data[campo_clave]==key,:]
                data.loc[data[campo_clave]==key, column] = fun(bloque)
                
    elif method==2:
        r = data.groupby(key_columns).apply(fun)
        if type(r)==pd.core.frame.DataFrame:
            r.columns = column
        else:
            r.name = column
        data = ajunta(data,r,campos1=key_columns,only_extend=True)
        
    elif method==3:
        if type(key_columns)==str:
            key_columns = [key_columns]

        groups = data.groupby(key_columns).apply(fun)
        data[column] = ""
        # Ver como evctorizar esto, apply es muy lenta
        map_columna(data, lambda r: groups.loc[tuple(r[key_columns])], column)

    if type(column) is not list:
        column = [column]
            
    if track_changes:
        Refrito.getRecalentador().añadirPaso(map_bloque, data_in, data, params)
    
    return data

def groupby(data, index, agg_fields=None, track_changes=True):
    """ Interfaz en ingles """
    return agrupa(data, index, agg_fields, track_changes)
    
def agrupa(data, index, campos_agg=None, track_changes=True):
    """ Equivalente a GROUP BY de SQL

        El formato de campos_agg puede ser:
        
            - Diccionario con nombre campo y operación de agrupación. Ejemplo:
              
               {"ORDEN":"count", ...}
              
              en este caso el campo agregado tiene igual nombre que el original
              
           -  Diccionario con nombre campo de salida y tupla con nombre campo 
              original y operación de agrupación. Ejemplo:
               
               {"PESO_TUTELA":("ORDEN","count"),...}

               en este caso se puede cambiar el nombre del campo de salida
               
           - None: hace sum con todos los campos menos los del index

        No se pueden mezclar
        
        Siempre devuelve un DataFrame nuevo
    """
    data_in = data
    params = {"index":index, "campos_agg":campos_agg}

    agregacion_con_nombre = False
    if not campos_agg:
        campos_agg = 'sum'
    elif type(campos_agg) == dict:
        primera_clave = list(campos_agg.keys())[0]
        agregacion_con_nombre = type(campos_agg[primera_clave])==tuple and len(campos_agg[primera_clave])==2

    if type(index)!=list:
        index = [index]

    if len(index)==0: # Sin caolumnas de agrupacion, se agrupa todo en un registro
        map_columna(data, lambda r:1,"AGRAGRAGR")
        index = ["AGRAGRAGR"]
        
    reset_campo_index = data.index.name in index
    if reset_campo_index:
        nombre_index = data.index.name
        
    if agregacion_con_nombre:
        data = data.groupby(index, dropna=False).agg(**campos_agg)
    else:
        data = data.groupby(index, dropna=False).agg(campos_agg)
        
    if data.columns.nlevels > 1:
        data.columns = ['_'.join(c) if type(campos_agg[c[0]])==list else c[0] for c in data.columns]
    
    data.reset_index(inplace=True)

    if reset_campo_index:
        data.set_index(nombre_index, inplace=True)

    if "AGRAGRAGR" == index[0]:
        data.drop("AGRAGRAGR", axis=1, inplace=True)

    if track_changes:
        Refrito.getRecalentador().añadirPaso(agrupa, data_in, data, params)
                
    return data

def pivot(data, index, columns, values, track_changes=True):
    """ Interfaz en ingles """
    return pivot(data, index, columns, values, track_changes)

def pivota(data, index, columns, values, track_changes=True):
    """ Hace un PIVOT (sin agrupar) 
            Con dataframe es directo
                            
        Devuelve un DataFrame
    """
    params = {"index":index, "columns":columns,"values":values}
    
    r = data.pivot(index=index, columns=columns, values=values)
    r.columns = [col[0] if col[1]=='' else col [1] for col in r.columns]
    r.reset_index(inplace=True)

    if track_changes:
        Refrito.getRecalentador().añadirPaso(pivota, data, r, params)            

    return r

def sort(data, by, inplace=True, kind='quicksort', track_changes=True):
    """ Interfaz en ingles """
    return ordena(data, by, inplace, kind, track_changes)
    
def ordena(data, by, inplace=True, kind='quicksort', ascending=True, key=None, track_changes=True):
    """
        Pues eso
    """
    data_in = data
    params = {"by":by, "kind":kind, "key":key, "inplace":inplace}
    
    if inplace:
        data.sort_values(by=by, kind=kind, ascending=ascending, key=key, inplace=True)
    else:
        data = data.sort_values(by=by, kind=kind, ascending=ascending, key=key, inplace=False)

    if track_changes:
        Refrito.getRecalentador().añadirPaso(ordena, data_in, data, params)
        
    return data

def concat(lista, not_repeated=True, use_field=None, keep='first', track_changes=True):
    """ Interfaz en ingles """
    return concatena(lista, not_repeated, use_field, keep, track_changes)
    
def concatena(lista, not_repeated=True, use_field=None, keep='first', track_changes=True):
    """
        Similar a una UNION de SQL
        
       Concatena una lista de DataFrames, sin repetir valores de índice
       
       se añade el primer data frame y a continuación se van añadiendo los 
       registros que no estén en los antriores, donde "que no estén" se 
       determina por el index del dataframe
       
    """
    params = {"not_repeated":not_repeated, "keep":keep, "use_field":use_field}
    
    madre = pd.concat(lista)
    
    if not_repeated:
        if use_field:
            madre.set_index(use_field, inplace=True)
    
        madre = madre[~madre.index.duplicated(keep=keep)]
    
        if use_field:
            madre.reset_index(inplace=True)

    if track_changes:
        Refrito.getRecalentador().añadirPaso(concatena, tuple(lista), madre, params,True)
        
    return madre

def histograma_temporal(data, campo_ini, campo_fin, campo_hist, inicio=None, fin=None, campo_peso=None):
    """
       Realiza un histograma continuo para el campo categoria dado por campo_hist
          donde cada registro de data inicia en el momento dado por campo campo_ini y 
          termina en el momento dado por campo_fin, cada registro cuenta como 1 excepto si
          se da un campo_peso que se asigna ese peso en el registro

          Los parámetros inicio y fin marcan el inicio y fin del histograma temporal

          la salida en un datafram con todas los momentos ordenados y consignados en 
          el índice del dataframe y en las diversas columnas la cantidad de cada categoria
          en cada instante

          (los campos inicio y fin no tendrías porqué ser fechas pero si marcar un orden secuencial)
    """
    if inicio == None:
        inicio = data[campo_ini].min()
        
    if fin == None:
        fin = data[campo_fin].max()
        
    data = filtra(data, lambda r: r[campo_ini]>fin or r[campo_fin]<inicio, inplace=False)

    # Extraigo categorias y se inicializan
    categorias = set(data[campo_hist])
    camposagg = {}
    data[list(categorias)] = 0
    for cat in categorias:
        map_columna(data, lambda row: (row[campo_peso] if campo_peso else 1) if row[campo_hist]==cat else 0, cat)
        camposagg[cat] = 'sum'

    #Agrupacion por inicio
    agrups_ini = agrupa(data, [campo_ini], camposagg)
    agrups_ini.set_index(campo_ini, inplace=True)
    if inicio not in agrups_ini.index: 
        agrups_ini.loc[inicio] = 0

    #Agrupacion por final
    agrups_fin = agrupa(data, [campo_fin], camposagg)
    map_columna(agrups_fin,lambda r:r[campo_fin] + pd.Timedelta(days=1),campo_fin)
    agrups_fin.set_index(campo_fin, inplace=True)
    if fin not in agrups_fin.index: 
        agrups_fin.loc[fin] = 0

    res = agrups_ini.add(-agrups_fin,fill_value=0).cumsum()
    res.index.name = 'index'
    filtra(res, lambda r: r['index']<inicio or r['index']>fin, inplace=True)
    
    return res
    
def ver_cambios(df0, df1, campo='DESC_CCE_1', clave='NIF', key=None):
    """ crea matriz de confusión con origen y destino de campo """
    c0 = campo + str(0)
    c1 = campo + str(1)

    df0 = selecciona(df0,[clave,campo],inplace=False)
    renombra_columnas(df0,[campo],[c0])

    df1 = selecciona(df1,[clave,campo],inplace=False)
    renombra_columnas(df1,[campo],[c1])

    cambios = ajunta(df0,df1,how='outer',campos1=clave)
    #cambios[c0] = cambios[c0].fillna('Entrada')
    #cambios[c1] = cambios[c1].fillna('Salida')
    map_columna(cambios, lambda r: 'Entrada' if pd.isnull(r[c0]) else r[c0], c0)
    map_columna(cambios, lambda r: 'Salida'  if pd.isnull(r[c1]) else r[c1], c1)
    #
    #cambios.reset_index(inplace=True)

    cc = agrupa(cambios,[c0,c1],{clave:'count'})
    cc = pivota(cc,[c0],[c1],[clave])
    #cc.fillna(0,inplace=True)
    map_columna(cc, lambda r: r.fillna(0), cc.columns)
    #
    renombra_columnas(cc,[c0],[campo])

    for val in set(cc[campo]):
        if val not in cc.columns:
            if val == 'Entrada': continue
            cc[val] = 0

    for val in set(cc.columns[1:]):
        if val not in set(cc[campo]):
            if val == 'Salida': continue
            cc.loc[-1] = [val] + [0]*(len(cc.columns)-1)

    if key is not None:
        ordena(cc,by=campo, key=key, inplace=True)
        #cc.sort_values(by=campo, key=key, inplace=True)
        lc = np.array(cc.columns[1:])
        cc = selecciona(cc, [campo] +  list(lc[np.argsort(key(lc))]), inplace=False, en_orden=True)

    #cc.reset_index(inplace=True, drop=True)

    return cc

##############################################################################################
########## Elementos para convertir listas de objetos en un dataframe ########################    
##############################################################################################
class ObjetoGenerico:
    
    def __init__(self, dictionary, date_format='%Y-%m-%d', convert_nums=True):
        #print(dictionary)
        self.date_format = date_format
        self.__dict__ = {k:self.convert(v,convert_nums) for k,v in dictionary.items()}
        
    def convert(self, value,convert_nums=True):
        if value is None: return value
        if type(value)==float or type(value)==int: return value

        if convert_nums:
            funs_c = [lambda v:datetime.strptime(v,self.date_format),int,float,str]
        else:
            funs_c = [lambda v:datetime.strptime(v,self.date_format),str]

        for fun_c in funs_c:
            try:
                value_c = fun_c(value)
                return value_c
            except:
                pass
            
        return value

def asignar_campo(objeto, campo, valor):
    """
        Asigna un valor a un atributo de un objeto
    """
    objeto.__dict__[campo] = valor

def objectos_to_dataframe(objects, campos=None, return_fields_only=False):
    """ Convierte un dict de clave-valor con objetos en los valores a un DAtaFrame
        Si no le pasas los campos, extrae todos los atributos de los objetos
            Si algún atributo es un dict, lo expande
            Si algún atributo es un objeto, no hace nada

        Si return_fields_only es True, devuelve la estructura de campos 
              que hubiera extraido
    """
    tipos = (int, float, str, bool, datetime)
    if not campos:
        campos = set()
        for k in objects: 
            obj = objects[k]
            for c1, v1 in obj.__dict__.items():
                if isinstance(v1, tipos):
                    campos.add(c1)
                elif isinstance(v1, dict):
                    for c2,v2 in v1.items():
                        if isinstance(v2, tipos) and isinstance(c2, tipos):
                            campos.add(c1 + "::" + str(c2))

    data = {campo:[] for campo in campos}
    
    if return_fields_only:
        return data
    
    index = []
    for key in objects:
        index.append(key)
        for campo in campos:
            if "::" in campo:
                #print(campo)
                c1, c2 = campo.split("::")
                #print(c1,c2)
                d = objects[key].__dict__[c1] if c1 in objects[key].__dict__ else None
                valor = d[c2] if d and c2 in d else float('nan')
            else:
                valor = objects[key].__dict__[campo]
                
            data[campo].append(valor)
            
    return pd.DataFrame(data=data, index=index).convert_dtypes() if data else pd.DataFrame()
    
##############################################################################################
##############################################################################################
##############################################################################################
