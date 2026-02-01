Grupo 3 - Sala 3
Integrantes: Rodrigo, Anna, Henrique, Vitor, Suellyn, Giselle e Felipe

A ideia do nosso projeto foi criar um modelo de difusão (o Mini-DDPM) do zero. Basicamente, ele é um modelo generativo que aprende a transformar ruído em imagens de dígitos (MNIST). 
O processo tem duas partes: na difusão direta, a gente vai "estragando" a imagem original com ruído até ela virar puro caos. Depois, treinamos uma rede U-Net para fazer o caminho inverso, aprendendo a limpar esse ruído passo a passo até reconstruir a imagem.
No fim, o modelo consegue criar números novos começando só do ruído.


QUESTIONÁRIO:
O que o beta_t faz?
Ele dita o ritmo da bagunça. Controla a quantidade exata de ruído que entra em cada etapa da difusão direta.

O que o alpha_bar_t representa?
Ele serve para mostrar o acumulado desse processo desde o começo até o tempo determinado.

Por que prever o epsilon (ruído)?
É mais fácil o modelo aprender a identificar o que é sujeira (ruído) para conseguir "subtrair" isso da imagem, deixando só o que importa.

Onde entra o "aprender a distribuição"?
Em modelos como esse, aprender a distribuição significa que a rede entendeu o padrão estatístico por trás das imagens de treino. Em vez de decorar os números, ela aprende a lógica de como os pixels se organizam. Assim, quando ela reverte o ruído, ela não está copiando, mas gerando um dado novo que faz sentido dentro daquele universo de dígitos manuscritos.


