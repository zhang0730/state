import sys
import h5py
import hydra
import logging
import numpy as np

from omegaconf import DictConfig

import torch
import lightning as L

from torch import nn
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss

sys.path.append('.')
from vci.data import MultiDatasetSentences, MultiDatasetSentenceCollator
from vci.model import LitUCEModel
from vci.train.trainer import get_ESM2_embeddings


log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="eval")
def main(cfg: DictConfig):
    checkpoint = '/checkpoint/ctc/ML/uce/model_checkpoints/exp_ds_medium_100_layers_4_dmodel_256_samples_1024_max_lr_0.0004_op_dim_256-epoch=6-step=82397.ckpt'

    model = LitUCEModel.load_from_checkpoint(checkpoint)
    all_pe = get_ESM2_embeddings(cfg)
    all_pe.requires_grad= False
    model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
    model.pe_embedding.to(model.device)
    model.binary_decoder.requires_grad= False
    model.eval()

    # Setup Data
    dataset = MultiDatasetSentences(cfg)
    multi_dataset_sentence_collator = MultiDatasetSentenceCollator(cfg)

    # Make the dataloader outside of the
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=multi_dataset_sentence_collator,
                            num_workers=1,
                            persistent_workers=True)

    embeddings = []
    losses = []
    f = None
    with torch.no_grad():
        i = 1
        criterion = BCEWithLogitsLoss()
        for i, batch in enumerate(dataloader):
            torch.cuda.empty_cache()

            batch_sentences = batch[0].to(model.device)
            mask = batch[1].to(model.device)
            X = batch[2].to(model.device)
            Y = batch[3].to(model.device).squeeze()
            dataset_nums = batch[5].to(model.device)

            batch_sentences = model.pe_embedding(
                batch_sentences.long())
            batch_sentences = nn.functional.normalize(batch_sentences, dim=2)
            gene_output, embedding = model(batch_sentences, mask=mask)

            embeddings.append(embedding.detach().cpu().numpy())

            dataset_num_emb = model.dataset_num_embedding(dataset_nums)
            X = model.pe_embedding(X.long())
            X = model.gene_embedding_layer(X)

            embedding = embedding.unsqueeze(1).repeat(1, X.shape[1], 1)
            dataset_num_emb = dataset_num_emb.unsqueeze(1).repeat(1, X.shape[1], 1)

            combine = torch.cat((X, embedding, dataset_num_emb), dim=2)
            decs = model.binary_decoder(combine).squeeze()

            loss = criterion(input=decs, target=Y)
            losses.append(loss.detach().cpu().numpy())

            log.info(f'new_batch_{i} {embedding.shape} {loss.shape} {loss}')
            i = i + 1

            if i % 100 == 0:
                embeddings = np.vstack(embeddings)
                losses = np.vstack(losses)
                log.info(f'Loss {losses.mean()}')
                if not f:
                    f = h5py.File('/checkpoint/ctc/ML/uce/embeddings.h5', 'a')
                    f.create_dataset('embedding',
                                     chunks=True,
                                     data=embeddings,
                                     maxshape=(None, 256))
                    f.create_dataset('loss',
                                     chunks=True,
                                     data=losses,
                                     maxshape=(None, 1))
                else:
                    f['embedding'].resize((f['embedding'].shape[0] + embeddings.shape[0]), axis=0)
                    f['embedding'][-embeddings.shape[0]:] = embeddings

                    f['loss'].resize((f['loss'].shape[0] + losses.shape[0]), axis=0)
                    f['loss'][-losses.shape[0]:] = losses

                embeddings = []
                losses = []


if __name__ == "__main__":
    main()
