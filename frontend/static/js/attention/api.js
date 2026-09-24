/**
 * Client of the attention JSON API. Responses are cached (as promises) so
 * moving a slider back and forth never downloads the same block twice.
 *
 * `familiesUrl` is `/api/jobs/<job>/attention/<run>/families/`; every other
 * route is relative to it.
 */
export class AttentionApi {
  constructor(familiesUrl) {
    this.familiesUrl = familiesUrl;
    this.cache = new Map();
  }

  /** `{ family: { shape, summary } }` for every extracted family. */
  async families() {
    const data = await this.#getJson(this.familiesUrl);
    return data ? data.families : {};
  }

  /** `{ weights, stats, links, contact_precision }` of one block. */
  block(family, block) {
    return this.#getJson(`${this.familiesUrl}${family}/blocks/${block}/`);
  }

  /** `{ shape, points: [[i, j, k, value], ...] }` above `threshold`. */
  cubePoints(family, block, threshold) {
    return this.#getJson(`${this.familiesUrl}${family}/blocks/${block}/cube/?threshold=${threshold.toFixed(2)}`);
  }

  #getJson(url) {
    if (!this.cache.has(url)) {
      const request = fetch(url).then((response) => (response.ok ? response.json() : null));
      request.then((data) => data === null && this.cache.delete(url));
      this.cache.set(url, request);
    }
    return this.cache.get(url);
  }
}
