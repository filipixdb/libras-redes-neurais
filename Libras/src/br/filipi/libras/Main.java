package br.filipi.libras;

public class Main {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		
		System.out.println("Lendo entradas...");
		try {
			Algoritmos.leEntradas();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		System.out.println("Processando entradas...");
		Algoritmos.processaEntradasSimples();
		Algoritmos.estendeEntradasProcessadasSimples();

//		System.out.println("testando matriz");
//		Algoritmos.testaMatriz();
		
//		System.out.println("Clusterizando com SOM...");
//		Algoritmos.clusterizaSOM();
		
//		System.out.println("Clusteriza com ART1...");
//		Algoritmos.clusterizaART1();
		
//		System.out.println("Clusteriza com KMeans");
//		Algoritmos.clusterizaKMeans();
		
		System.out.println("Processando entradas SDM");
		Algoritmos.processaEntradasSDMMatriz();
		System.out.println("Grava com SDM");
		Algoritmos.constroiSDM();
//		Algoritmos.constroiSDMEnderecoIgualPalavra();
	}

}
