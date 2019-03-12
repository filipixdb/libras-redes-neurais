package br.filipi.libras;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;

import org.encog.Encog;
import org.encog.mathutil.matrices.BiPolarUtil;
import org.encog.ml.MLCluster;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.data.specific.BiPolarNeuralData;
import org.encog.ml.kmeans.KMeansClustering;
import org.encog.neural.art.ART1;
import org.encog.neural.som.SOM;
import org.encog.neural.som.training.basic.BasicTrainSOM;
import org.encog.neural.som.training.basic.neighborhood.NeighborhoodSingle;
import org.encog.util.arrayutil.NormalizeArray;
import org.encog.util.kmeans.KMeansUtil;

import cern.colt.bitvector.BitMatrix;
import cern.colt.bitvector.BitVector;

import edu.memphis.ccrg.lida.episodicmemory.sdm.BitVectorUtils;
import edu.memphis.ccrg.lida.episodicmemory.sdm.SparseDistributedMemory;
import edu.memphis.ccrg.lida.episodicmemory.sdm.SparseDistributedMemoryImpl;

public class Algoritmos {
	
	private static ArrayList<ArrayList<Double>> entradasLidas = null;
	private static double[][] entradasProcessadas = null;
	private static BitVector[] entradasProcessadasSDM = null;
	
	private static ArrayList<Double> arrayClasses = null;
	
	private static int tamanhoMatriz = 15;
	
	public static void leEntradas() throws Exception{
		
		ArrayList<ArrayList<Double>> dados = new ArrayList<ArrayList<Double>>();
		
		//ler entradas do dataset
		BufferedReader br = new BufferedReader(new FileReader("movement_libras.data.txt"));
		   try {
		       String line = br.readLine();
		        while (line != null) {
		        	//aqui processar a linha
		        	ArrayList<Double> instancia = new ArrayList<>();
		        	for (String s : line.split(",")){
		        		Double d = new Double(s);
		        		instancia.add(d);
		        	}
		        	dados.add(instancia);
		        	
		            line = br.readLine();
		        }
		        
		    } finally {
		        br.close();
		    }	
		
		   
		   //embaralhar as instancias
//		   Collections.shuffle(dados);
		   
		   //preencher o mapa que associa entradas a classes
		   arrayClasses = new ArrayList<>();
		   for (int x = 0; x < dados.size(); x++){
			   arrayClasses.add(dados.get(x).get(90));
			}
		   
		   entradasLidas = dados;
	}
	
	
	
	public static void processaEntradasSimples(){
//		   Antigo sem usar as transicoes (as is)
		   
		   //converter os dados para double[][]
		   
		   double[][] dadosfinal = new double[entradasLidas.size()][entradasLidas.get(0).size()-1];
		   
		   for (int i = 0; i < entradasLidas.size(); i++){
			   
			   for (int j = 0; j < entradasLidas.get(0).size()-1; j++){//desprezar o ultimo atributo (classe)
				   
				   dadosfinal[i][j] = entradasLidas.get(i).get(j);
			   }
			   
		   }
		   
		   entradasProcessadas = dadosfinal;
		
	}
	
	
	
	public static void processaEntradasDiferenca(){
//		---- Antigo processa diferencas e direcoes		   
		   //fazer agora o array com as transições
		   
		   double[][] dadosTransicao = new double[entradasLidas.size()][entradasLidas.get(0).size()-1-2];
		   
		   
		   for (int inst = 0; inst < entradasLidas.size(); inst++){
			   
			   double mediaDiferencas = 0;
			   double desvioPadraoDiferencas = 0;
			   
			   //calcular a media das diferencas pra usar adiante
			   for (int trans = 0; trans < entradasLidas.get(0).size()-1-2; trans++){
				   mediaDiferencas += entradasLidas.get(inst).get(trans+2) - entradasLidas.get(inst).get(trans);
			   }
			   mediaDiferencas = mediaDiferencas/(entradasLidas.get(0).size()-1-2);
			   
			   
			   //calcular o desvio padrao para usar mais adiante
			   for (int trans = 0; trans < entradasLidas.get(0).size()-1-2; trans++){
				   double dif = entradasLidas.get(inst).get(trans+2) - entradasLidas.get(inst).get(trans);
				   desvioPadraoDiferencas += Math.pow(dif - mediaDiferencas, 2);
			   }
			   desvioPadraoDiferencas = desvioPadraoDiferencas/(entradasLidas.get(0).size()-1-2);
			   
			   
			   //definir limiar para usar adiante
			   double limiar = Math.abs(Math.abs(mediaDiferencas) - 0.7*desvioPadraoDiferencas);
			   
			   System.out.println("Limiar: " + limiar + " - Media: " + mediaDiferencas + " - Desvio: "+ desvioPadraoDiferencas);
			   
			   for (int trans = 0; trans < entradasLidas.get(0).size()-1-2; trans++){
				   
				   double diferenca =  entradasLidas.get(inst).get(trans+2) - entradasLidas.get(inst).get(trans);
				   
				   if (diferenca > limiar){
					   dadosTransicao[inst][trans] = 1;
				   }else if (diferenca < -limiar){
					   dadosTransicao[inst][trans] = -1;
				   }else{
					   dadosTransicao[inst][trans] = 0;
				   }
				   
				   //colocar pra fazer apenas a diferenca (sem direcao)
				   dadosTransicao[inst][trans] = diferenca;
				   
//				   if (inst == 289){
//					   System.out.println(dadosTransicao[inst][trans] + " - diferenca: " + diferenca);
//					   System.out.print(Math.abs(diferenca) + ",");
//				   }
				   
			   }
			   
		   }
		   
		   entradasProcessadas = dadosTransicao;


	}
	
	
	
	public static void processaEntradasSegmento(){
//		Dividir cada instancia em 9 segmentos de 5 posicoes
		
		double[][] dadosSegmentos = new double[entradasLidas.size()][9*2];
		
		
		for (int inst = 0; inst < entradasLidas.size(); inst++){
			
			
			for (int seg = 0; seg < (9); seg++){
				
				double diferencaX = entradasLidas.get(inst).get((seg*10)+8) - entradasLidas.get(inst).get(seg*10);
				dadosSegmentos[inst][seg*2] = diferencaX;
				
				double diferencaY = entradasLidas.get(inst).get((seg*10)+8+1) - entradasLidas.get(inst).get((seg*10)+1);
				dadosSegmentos[inst][(seg*2)+1] = diferencaY;
				
				if (inst == 337){
					
					System.out.println("Segmento " + seg + " X1: " + entradasLidas.get(inst).get(seg*10) + " - X2: " + entradasLidas.get(inst).get((seg*10)+8));
					System.out.println("Segmento " + seg + " Y1: " + entradasLidas.get(inst).get((seg*10)+1) + " - Y2: " + entradasLidas.get(inst).get((seg*10)+8+1));
					
					
					System.out.println("Segmento " + seg + " - Dif X: " + diferencaX + " - Dif Y: " + diferencaY);
				}
			}
			
		}
		
		entradasProcessadas = dadosSegmentos;
	}
	
	
	public static void processaEntradasSegmentoDirecao(){
//		Dividir cada instancia em 9 segmentos de 5 posicoes
		
		double[][] dadosSegmentos = new double[entradasLidas.size()][9*2];
		
		
		for (int inst = 0; inst < entradasLidas.size(); inst++){
			
			double[] diferencasX = new double[9];
			double[] diferencasY = new double[9];
			
					
			for (int seg = 0; seg < (9); seg++){
				
				double diferencaX = entradasLidas.get(inst).get((seg*10)+8) - entradasLidas.get(inst).get(seg*10);
				diferencasX[seg] = diferencaX;
				
				double diferencaY = entradasLidas.get(inst).get((seg*10)+8+1) - entradasLidas.get(inst).get((seg*10)+1);
				diferencasY[seg] = diferencaY;
		
			}
			
			
			double limiarX = calculaLimiar(diferencasX);
			double limiarY = calculaLimiar(diferencasY);
			
			
//			Usar o maior limiar
//			if (limiarX > limiarY){
//				limiarY = limiarX;
//			}else if(limiarY > limiarX){
//				limiarX = limiarY;
//			}

			
//			Usar media dos limiares
			limiarX = (limiarX+limiarY)/2;
			limiarY = limiarX;
			
			
			//calcular direcoes usando o limiar
			for (int seg = 0; seg < (9); seg++){
				
				double direcaoX;
				if (diferencasX[seg] > limiarX){
					direcaoX = 1;
				}else if (diferencasX[seg] < -limiarX){
					direcaoX = -1;
				}else{
					direcaoX = 0;
				}
				dadosSegmentos[inst][seg*2] = direcaoX;
				
				
				
				double direcaoY;
				if (diferencasY[seg] > limiarY){
					direcaoY = 1;
				}else if (diferencasY[seg] < -limiarY){
					direcaoY = -1;
				}else{
					direcaoY = 0;
				}
				dadosSegmentos[inst][(seg*2)+1] = direcaoY;
				
		
				
				if (inst == 48){
					System.out.println("Segmento " + seg + " X1: " + entradasLidas.get(inst).get(seg*10) + " - X2: " + entradasLidas.get(inst).get((seg*10)+8) + " - DirX: " + direcaoX);
					System.out.println("Segmento " + seg + " Y1: " + entradasLidas.get(inst).get((seg*10)+1) + " - Y2: " + entradasLidas.get(inst).get((seg*10)+8+1) + " - DirY: " + direcaoY);
				}
				
			
			}
			
			
		}
		
		entradasProcessadas = dadosSegmentos;
	}
	
	
	private static double calculaLimiar(double[] valores){
		
	   double media = 0;
	   double desvioPadrao = 0;
	   
	   //calcular a media dos modulos pra usar adiante
	   for (int seg = 0; seg < (valores.length); seg++){
			media+=Math.abs(valores[seg]);
		}
	   media = media/valores.length;
	   
	   //calcular o desvio padrao para usar mais adiante
	   for (int trans = 0; trans < valores.length; trans++){
		   desvioPadrao += Math.pow(Math.abs(valores[trans]) - media, 2);
	   }
	   desvioPadrao = desvioPadrao/(valores.length);
	   
	   //definir limiar
	   double limiar = 0.5*media;// - 2.0*desvioPadrao;
	   
//	   System.out.println("Limiar: " + limiar + " -- Media: " + media + " -- Desvio: "+ desvioPadrao);
		 
	   return limiar;
	}
	
	
	
	
	public static void clusterizaSOM(){
		
		double[][] dados = entradasProcessadas;
		
		MLDataSet dadosTreinamento = new BasicMLDataSet(dados, null);
		
		SOM som = new SOM(dados[0].length, 15);
		
		som.reset();
		
		BasicTrainSOM treino = new BasicTrainSOM(som, 0.1, dadosTreinamento, new NeighborhoodSingle());
		
		int iteracao = 0;
		
		for(iteracao = 0; iteracao <= 360; iteracao+=1)
		{
			treino.iteration();
//			System.out.println("Iteracao: " + iteracao + ", Error:" + treino.getError());
		}
		
		//testando classificacao
		
		int[][] quantidadesPorCluster = new int[15][15];
		
		for (int k = 0; k < 360; k += 1){
			MLData instancia = new BasicMLData(dados[k]);
			int resp = som.classify(instancia);
			
			quantidadesPorCluster[resp][arrayClasses.get(k).intValue() -1] += 1;
			
//			System.out.println("Classificacao: "+ som.classify(instancia) + " -- Correto: " + arrayClasses.get(k) );
		}

		System.out.println("printando matriz respostas..");
		for (int z = 0; z < 15; z++){
			System.out.println("Classe resposta " + z);
			for (int z2 = 0; z2 <15; z2++){
				System.out.print(quantidadesPorCluster[z][z2] + " - ");
			}
			System.out.println("\n");
		}
		
//		for (int z = 0; z < 15; z++){
//			System.out.print(z);
//			for (int z2 = 0; z2 <15; z2++){
//				System.out.print( " & " + quantidadesPorCluster[z][z2]);
//			}
//			System.out.println("\\\\ \\hline");
//		}
//			
		
		
		
	}
	
	
	
	public static void clusterizaART1(){
		
		double[][] dados = entradasProcessadas;
		
		
		ART1 art1 = new ART1(dados[0].length, 15);
		
		
		for (int i = 0; i < 360; i++){

			BiPolarNeuralData in = new BiPolarNeuralData(BiPolarUtil.double2bipolar(dados[i]));
//			System.out.println(in);
			
			BiPolarNeuralData out = new BiPolarNeuralData(15);
			art1.compute(in, out);
			
//			System.out.println(out);
			
			if (art1.hasWinner()){
				System.out.println("Instancia " + i + " -- Classe " + art1.getWinner());
			}else{
				System.out.println("Instancia " + i + " -- Classe indefinida");
			}
		}
		
		//testando classificacao
//		for (int k = 0; k < 360; k += 24){
//			MLData instancia = new BasicMLData(dados[k]);
//			System.out.println("Classe "+ art1.classify(instancia) + " - Padrao " + k );
//		}
		
	}

	
	
	public static void clusterizaKMeans(){
		
		double[][] dados = entradasProcessadas;
		
		HashMap<MLData, Integer> mapaDataPraClasse = new HashMap<MLData, Integer>();
		
		
		BasicMLDataSet dataset = new BasicMLDataSet();

		for (double[] element : dados) {
			dataset.add(new BasicMLData(element));
		}

		
		//associar as classes corretas
		for (int aux = 0; aux < 360; aux++){
			MLDataPair data = dataset.getData().get(aux);
			mapaDataPraClasse.put(data.getInput(), arrayClasses.get(aux).intValue()-1);
		}
		
		
		
		KMeansClustering kmeans = new KMeansClustering(15, dataset);

		kmeans.iteration(500);
		
		int i = 1;
		for (MLCluster cluster : kmeans.getClusters()) {
			final MLDataSet ds = cluster.createDataSet();
			System.out.println("Cluster " + (i++) + " -- #Instancias: " + ds.getRecordCount());
		}
		
		
		
		System.out.println("Printando resultados kmeans...");
		int[][] quantidadesPorCluster = new int[15][15];
		int indiceCluster = 0;
		for (MLCluster cluster : kmeans.getClusters()){
			for (MLData d : cluster.getData()){
				int indiceClasse = mapaDataPraClasse.get(d).intValue();
				quantidadesPorCluster[indiceCluster][indiceClasse] += 1;

			}
			indiceCluster++;
		}
			
//		for (int z = 0; z < 15; z++){
//			System.out.println("Classe resposta " + z);
//			for (int z2 = 0; z2 <15; z2++){
//				System.out.print(quantidadesPorCluster[z][z2] + " - ");
//			}
//			System.out.println("\n");
//		}
		
		for (int z = 0; z < 15; z++){
			System.out.print(z);
			for (int z2 = 0; z2 <15; z2++){
				System.out.print( " & " + quantidadesPorCluster[z][z2]);
			}
			System.out.println(" \\hline");
		}
		
		
	}
	
	
	
	//// SDM
	
	public static void processaEntradasSDM(){
		
		double[][] dados = entradasProcessadas;
		
		//converter para binario considerando que tenho as transicoes
		
		BitVector[] bitvectors = new BitVector[dados.length];
		
		for (int i = 0; i < dados.length; i++){
			bitvectors[i] = new BitVector(dados[0].length);
			
			for (int j = 0; j < dados[0].length; j++){
				
				if (dados[i][j] > 0){
					bitvectors[i].put(j, true);
				}else{
					bitvectors[i].put(j, false);
				}
			}
		}
		
		entradasProcessadasSDM = bitvectors;
		
		
	}
	
	
	public static void constroiSDM(){
		
		BitVector[] dados = entradasProcessadasSDM;
		
		SparseDistributedMemoryImpl sdm = new SparseDistributedMemoryImpl(10000, 86, 15, dados[0].size());
		
		for (int i = 0;  i < dados.length; i+=2){
			
			BitVector vetorClasseCorreta = new BitVector(15);
			int classeCorreta = (int) Math.round(arrayClasses.get(i)-1.0);
			for (int bitClasse = 0; bitClasse < 15; bitClasse++){
				if (classeCorreta == bitClasse){
					vetorClasseCorreta.put(bitClasse, true);
				}else{
					vetorClasseCorreta.put(bitClasse, false);
				}
			}
			
			sdm.store(vetorClasseCorreta , dados[i]);
		}
		
		for (int out = 1; out < dados.length; out+=2){
			System.out.println("Retornando dados sdm: instancia: " + out + " -- classe correta: " + (arrayClasses.get(out)-1) + " -- classe resposta: " +  sdm.retrieve(dados[out]) );
			
		}
		
		
//		for (int classe = 0 ; classe < 15; classe++){
//			System.out.println("Distancias para um membro da classe " + classe);
//			for (int out = 0; out < dados.length; out++){
//				System.out.println("instancia " + out + " -- " +   BitVectorUtils.hamming(dados[classe*24], dados[out]));//       sdm.retrieve(dados[out]) );
//				
//			}
//		}
		
		
	}
	
	public static void constroiSDMEnderecoIgualPalavra(){
		
		BitVector[] dados = entradasProcessadasSDM;
		
		SparseDistributedMemoryImpl sdm = new SparseDistributedMemoryImpl(10000, 89, dados[0].size());
		
		for (int i = 0;  i < dados.length; i+=1){
			
			sdm.store(dados[i]);
		}
		
		
//		for (int out = 1; out < dados.length; out+=2){
//			System.out.println("Retornando dados sdm: instancia: " + out );
//			System.out.println(" -- entrada: " + dados[out]);
//			System.out.println(" -- resposta: " +  sdm.retrieve(dados[out]));
//		}
		
		
//		for (int classe = 0 ; classe < 15; classe++){
//			System.out.println("Distancias para um membro da classe " + classe);
//			for (int out = 0; out < dados.length; out++){
//				System.out.println("instancia " + out + " -- " +   BitVectorUtils.hamming(dados[classe*24], dados[out]));//       sdm.retrieve(dados[out]) );
//				
//			}
//		}

		
//		Exibir repostas como matriz
		for (int out = 1; out < dados.length; out+=6){
			System.out.println("Instancia: " + out );
			BitVector resposta = sdm.retrieveIterating(dados[out]);
			
			for (int linha = 0; linha < tamanhoMatriz; linha++){
				
				for (int coluna = 0; coluna < tamanhoMatriz; coluna++){
					if (dados[out].get((linha*tamanhoMatriz)+coluna)){
						System.out.print("#");
					}else{
						System.out.print("_");
					}
				}
				
				System.out.print("     ");
				
				for (int coluna = 0; coluna < tamanhoMatriz; coluna++){
					if (resposta.get((linha*tamanhoMatriz)+coluna)){
						System.out.print("#");
					}else{
						System.out.print("_");
					}
				}
				
				System.out.println("");
				
			}
			System.out.println("\n");
		}

//		//imprime as distancias
//		for (int classe = 0 ; classe < 15; classe++){
//		System.out.println("Distancias para um membro da classe " + classe);
//		for (int out = 0; out < dados.length; out++){
//			System.out.println("instancia " + out + " -- " +   BitVectorUtils.hamming(dados[classe*24], dados[out]));//       sdm.retrieve(dados[out]) );
//			}
//		}

		
	}
		
	
	
	private static BitMatrix representaMatriz(double[]pontos){
		
		//descobrir inicial e final
		double menorX = 1;
		double menorY = 1;
		double maiorX = 0;
		double maiorY = 0;
		
		for (int i = 0; i < pontos.length; i+=2){
			if (pontos[i] < menorX) menorX = pontos[i];
			if (pontos[i] > maiorX) maiorX = pontos[i];
			if (pontos[i+1] < menorY) menorY = pontos[i+1];
			if (pontos[i+1] > maiorY) maiorY = pontos[i+1];
		}
		maiorX = maiorX + 0.0000001;
		maiorY = maiorY + 0.0000001;
		
		//calcular tamanho do passo; descobrir maior direcao
//		System.out.println("Diferencas: " + (maiorX - menorX) + " , " + (maiorY-menorY));
		double passoX = ((maiorX-menorX)/tamanhoMatriz);
		double passoY = ((maiorY-menorY)/tamanhoMatriz);
		if (passoX > (4*passoY)){
			passoY = passoX;
			menorY = menorY - ((passoX*tamanhoMatriz)/2);
		}else if (passoY > (4*passoX)){
			passoX = passoY;
			menorX = menorX - ((passoY*tamanhoMatriz)/2);
		}
		
		//preencher a matriz; iterar nos pontos descobrindo onde eles se encontram
		BitMatrix matriz = new BitMatrix(tamanhoMatriz, tamanhoMatriz);
		matriz.clear();
		for (int j = 0; j<pontos.length; j+=2){
			int coordenadaX = (int) Math.floor((pontos[j]-menorX) / passoX);
			int coordenadaY = (int) Math.floor((pontos[j+1]-menorY) / passoY);
			matriz.put(coordenadaX, coordenadaY, true);
		}
		
		return matriz;
	}
	
	
	public static void testaMatriz(){
		BitMatrix matriz = representaMatriz(entradasProcessadas[0]);
		for (int linha = 0; linha< tamanhoMatriz; linha++){
			for (int coluna = 0; coluna < tamanhoMatriz; coluna++){
				if (matriz.get(coluna, linha)){
					System.out.print("##");
				}else{
					System.out.print("__");
				}
			}
			System.out.println("");
		}
		
	}
	
	public static void processaEntradasSDMMatriz(){
		double[][]dados = entradasProcessadas;
		
		BitVector[] matrizes = new BitVector[dados.length];
		
		for (int i = 0; i < dados.length; i++){
			matrizes[i] = representaMatriz(dados[i]).toBitVector();
		}
		
		entradasProcessadasSDM = matrizes;
	}
	
	public static void estendeEntradasProcessadasSimples(){
		
		double[][] dados = entradasProcessadas;
		
		double[][] dadosEstendidos = new double[dados.length][(dados[0].length * 2) -2];
		
		for (int i = 0; i< dados.length; i++){
			for (int j = 0; j< 90; j+=2){
				dadosEstendidos[i][j*2] = dados[i][j];//x original
				dadosEstendidos[i][(j*2)+1] = dados[i][j+1];//y original
				
				if (j<88){
					dadosEstendidos[i][(j+1)*2] = (dados[i][j+2] + dados[i][j])/2;//x intermediario
					dadosEstendidos[i][((j+1)*2)+1] = (dados[i][j+2+1] + dados[i][j+1])/2;//x intermediario
				}
			}
		}
		entradasProcessadas = dadosEstendidos;
	}
	
	
	
}
