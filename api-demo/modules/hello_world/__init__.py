from flask import Response
from stockfish import Stockfish
import json
import torch 
import numpy as np 
import math
import chess

def get_moves_dict():
    letters=list('abcdefgh')
    pawn_crown=[]
    pieces=list('qnrb')
    for x in range(len(letters)):
        for piece in pieces:
            if letters[x]== 'a':
                pawn_crown.append(letters[x]+'2'+letters[x]+'1'+piece)
                pawn_crown.append(letters[x]+'2'+letters[x+1]+'1'+piece)
                pawn_crown.append(letters[x]+'7'+letters[x]+'8'+piece)
                pawn_crown.append(letters[x]+'7'+letters[x+1]+'8'+piece)
            elif letters[x]== 'h':

                pawn_crown.append(letters[x]+'2'+letters[x]+'1'+piece)
                pawn_crown.append(letters[x]+'2'+letters[x-1]+'1'+piece)
                pawn_crown.append(letters[x]+'7'+letters[x]+'8'+piece)
                pawn_crown.append(letters[x]+'7'+letters[x-1]+'8'+piece)

            else:
                pawn_crown.append(letters[x]+'2'+letters[x]+'1'+piece)
                pawn_crown.append(letters[x]+'2'+letters[x-1]+'1'+piece)
                pawn_crown.append(letters[x]+'2'+letters[x+1]+'1'+piece)
                pawn_crown.append(letters[x]+'7'+letters[x]+'8'+piece)
                pawn_crown.append(letters[x]+'7'+letters[x-1]+'8'+piece)
                pawn_crown.append(letters[x]+'7'+letters[x+1]+'8'+piece)
                
    board_spaces=[x+str(y) for x in letters for y in range(1,9)]
    possible_moves=[x+str(y) for x in board_spaces for y in board_spaces]+pawn_crown
    moves_dict={ key:label for label,key in enumerate(possible_moves)}
    moves_dict['start']=len(moves_dict.values())
    moves_dict['end']=len(moves_dict.values())
    inv_moves_dict = {v: k for k, v in moves_dict.items()}
    return moves_dict,inv_moves_dict 
class Tokenizer:
    def __init__(self,path=None, max_length=76,vocab_size=323):
        self.max_length=max_length
        self.dictionary_board= {
            "/":0, #change value, and keep zero to mask
            "r":9,
            "n":10,
            "b":11,
            "q":12,
            "k":13,
            "p":14,
            "R":15,
            "N":16,
            "B":17,
            "Q":18,
            "K":19,
            "P":20,
            " ":321,
            "<PAD>":322
        }
        self.detokenizer_dict={v: k for k, v in self.dictionary_board.items()}
        self.dictionary_active_color={
            "w":21,
            "b":22
        }
        self.detokenizer_dict.update({v: k for k, v in self.dictionary_active_color.items()})
        self.dictionary_castling_rights={
            "K":23,
            "Q":24,
            "k":25,
            "q":26,
            "-":27
        }
        self.detokenizer_dict.update({v: k for k, v in self.dictionary_castling_rights.items()})
        self.dictionary_en_passant_square={
            "-":28,
            "a3":29,
            "b3":30,
            "c3":31,
            "d3":32,
            "e3":33,
            "f3":34,
            "g3":35,
            "h3":36,
            "a6":37,
            "b6":38,
            "c6":39,
            "d6":40,
            "e6":41,
            "f6":42,
            "g6":43,
            "h6":44
        }
        self.detokenizer_dict.update({v: k for k, v in self.dictionary_en_passant_square.items()})
        self.dictionary_half_move_clock={
        }
        self.dictionary_full_move_number={}
        
        self.vocab_size=vocab_size
        
        
    def board_to_token_vector(self, board):
        #print(board)
        #takes every charater on the board and returns a token vector
        return [ int(char) if char.isnumeric() else self.dictionary_board[char] for char in board ]
    
    def tokenize(self, fen):
        token_vector =[]
        fen_components=fen.split(" ")

        token_vector+= self.board_to_token_vector(fen_components[0])
        token_vector+=[self.dictionary_board[" "]]
        token_vector+= [self.dictionary_active_color[fen_components[1]]] # tokenize the active color
        token_vector+=[self.dictionary_board[" "]]
        token_vector+= [self.dictionary_castling_rights[char] for char in fen_components[2] ] # tokenize the castling rights
        token_vector+=[self.dictionary_board[" "]]
        token_vector+= [self.dictionary_en_passant_square[fen_components[3]]] # tokenize the en passant square
        token_vector+=[self.dictionary_board[" "]]
        token_vector+= [int(fen_components[4])+ 45] # tokenize the half move clock
        token_vector+=[self.dictionary_board[" "]]
        token_vector+= [int(fen_components[5])+ 45+100] # tokenize the full move number
        if(len(token_vector)<self.max_length):
            token_vector+=[self.dictionary_board["<PAD>"]]*(self.max_length-len(token_vector))
        else:
            print("bigger:", len(token_vector))
            
    
        return torch.LongTensor(token_vector)
    def detokenize(self,token_vector):
        fen=[]
        for token in token_vector:
            token=token.item()
            if token in range(1,9):
                fen.append(str(token))
            elif token > 44 and token<45+100:
                fen.append(str(token- 45))
            elif token >= 45+100 and token<self.dictionary_board[" "]:
                fen.append(str(token-(45+100)))
            else:
                fen.append(self.detokenizer_dict[token])
        return ''.join(fen).replace("<PAD>", "")

## NN MODEL

class PositionalEncoding(torch.nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class TransformerClassifier(torch.nn.Module):

    def __init__(self, vocab_size_fens,n_moves, dropout=0.1, d_model=512, n_labels=5, nhead=8, num_encoder_layers=4, dim_feedforward=2048):

        super(TransformerClassifier, self).__init__()
        self.d_model=d_model
        self.embedding_fens = torch.nn.Embedding(vocab_size_fens, d_model, padding_idx=322)
        self.embedding_moves = torch.nn.Embedding(n_moves, d_model, padding_idx=4273)
        self.pos_encoder_fens = PositionalEncoding(d_model, dropout)
        self.pos_encoder_moves = PositionalEncoding(d_model, dropout)
        self.transformer_model = torch.nn.Transformer(nhead=nhead, num_encoder_layers=num_encoder_layers
                                                ,num_decoder_layers=num_encoder_layers,dim_feedforward=dim_feedforward,
                                               d_model=d_model,batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.reduce_dim=torch.nn.Embedding(d_model,n_moves)
        self.linear = torch.nn.Linear(d_model, n_labels)
        self.relu = torch.nn.ReLU()

    def forward(self, fens_vector, mask_fens, moves_vector,mask_moves):
        embeded_fens=self.embedding_fens(fens_vector)* np.sqrt(self.d_model)
        fens_encoded=self.pos_encoder_fens(embeded_fens)
        embeded_moves=self.embedding_moves(moves_vector)* np.sqrt(self.d_model)
        moves_encoded=self.pos_encoder_moves(embeded_moves)
        pooled_output = self.transformer_model(src=fens_encoded,tgt=moves_encoded,
        src_key_padding_mask=mask_fens,tgt_key_padding_mask=mask_moves)
        dropout_output = self.dropout(torch.norm(pooled_output,dim=1))
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer

###### Variable and classes defined in initialization 

stockfish = Stockfish()
tokenizer=Tokenizer()

def hello_world(fen_postion):

    MAX_LENGTH_VALID_MOVES=100


    # set initial position from FEN 
    stockfish.set_fen_position(fen_postion)
    moves_dict,inv_moves_dict = get_moves_dict()
    tokenized_fen=tokenizer.tokenize(fen_postion)

    ## we create the model 
    transformer=TransformerClassifier(vocab_size_fens=tokenizer.vocab_size,n_moves=len(moves_dict.keys()),
                                  #d_model=(len(moves_dict.keys()) -2)//2, 
                                  n_labels=len(moves_dict.keys()), dim_feedforward=(len(moves_dict.keys())-2))
    device = torch.device("cpu")
    transformer.to(device)

    transformer.load_state_dict(torch.load('/Users/nicolasfelipevergaraduran/Documents/Computacion/NetworkAndesIA/RL_chess_engine/src/models/data_scientist/chess_transformer_v2.pth',map_location=device))
    transformer.eval()

    ##Let's play 
    
    with torch.no_grad():
        ##Load the data 

        batch_positions = tokenized_fen
        mask_batch_positions=torch.BoolTensor([True if token ==322 else False for token in tokenized_fen])

        ## Let's compute the legal moves for the position 
        board = chess.Board()
        board.set_fen(fen=fen_postion)
        legal_moves=[move.uci() for move in board.legal_moves]
        legal_moves=legal_moves+['end']*(MAX_LENGTH_VALID_MOVES-len(legal_moves))
        target_input=torch.LongTensor([moves_dict[move] for move in legal_moves])
        mask_target_input=torch.BoolTensor([True if token ==4273 else False for token in target_input])


        target_input =target_input.to(device)
        target_input =torch.reshape(target_input,(1,target_input.size()[0]))
        mask_target = mask_target_input.to(device)
        mask_target =torch.reshape(mask_target,(1,mask_target.size()[0]))
        mask_input = mask_batch_positions.to(device)
        mask_input =torch.reshape(mask_input,(1,mask_input.size()[0]))
        input_id = batch_positions.to(device)
        input_id=torch.reshape(input_id,(1,input_id.size()[0]))
        output=transformer(input_id , mask_input,target_input,mask_target)

        next_move=inv_moves_dict[torch.argmax(output).item()]
        
        if next_move not in legal_moves:
            next_move=stockfish.get_best_move() 

    return Response(json.dumps({'next_move':next_move, 'fen tokenized':tokenizer.detokenize(tokenizer.tokenize(fen_postion))}), mimetype='application/json')

