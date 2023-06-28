# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

<!-- insertion marker -->
## [v0.5.0](https://gitlab.com/ydethe/blocksim/tags/v0.5.0) - 2023-06-28

<small>[Compare with v0.4.0](https://gitlab.com/ydethe/blocksim/compare/v0.4.0...v0.5.0)</small>

### Added

- Added hypothesis tests ([cef5881](https://gitlab.com/ydethe/blocksim/commit/cef58810bc023a9d6f2d1e73b3a93a56fa86a700) by Yann BLAUDIN DE THE).

## [v0.4.0](https://gitlab.com/ydethe/blocksim/tags/v0.4.0) - 2023-06-28

<small>[Compare with v0.3.1](https://gitlab.com/ydethe/blocksim/compare/v0.3.1...v0.4.0)</small>

### Added

- Added naming of Plottables and pick event handling by default that prints the name of the Plottable. Removed color attribute in Trajectory class ([477b45f](https://gitlab.com/ydethe/blocksim/commit/477b45f2da84b8e6b88c2a85aacae79444fdca10) by Yann BLAUDIN DE THE).
- Added link under coverage badge ([1833569](https://gitlab.com/ydethe/blocksim/commit/18335690bb16e37d13711d5f190bfca375b2ece4) by Yann BLAUDIN DE THE).

### Fixed

- Fixed doc generation ([f2f8aa9](https://gitlab.com/ydethe/blocksim/commit/f2f8aa93183151e89977bd31fbc123002875301b) by Yann BLAUDIN DE THE).

### Removed

- Removed pdm.lock ([381ed4a](https://gitlab.com/ydethe/blocksim/commit/381ed4a27dc7b7c431bb6d0fad3290b4f12c399d) by Yann BLAUDIN DE THE).

## [v0.3.1](https://gitlab.com/ydethe/blocksim/tags/v0.3.1) - 2023-04-27

<small>[Compare with v0.3.0](https://gitlab.com/ydethe/blocksim/compare/v0.3.0...v0.3.1)</small>

### Added

- Added conditional deploy (on tag) ([0847f8e](https://gitlab.com/ydethe/blocksim/commit/0847f8e89cf760afaed9f432db61ea35a72a5afa) by Yann BLAUDIN DE THE).

## [v0.3.0](https://gitlab.com/ydethe/blocksim/tags/v0.3.0) - 2023-04-26

<small>[Compare with v0.2.0](https://gitlab.com/ydethe/blocksim/compare/v0.2.0...v0.3.0)</small>

### Added

- Added pdm-copier commit hook and updated gitlab CI ([e8c79d4](https://gitlab.com/ydethe/blocksim/commit/e8c79d443b8824c54e114a359a9279361dd3e5b0) by Yann BLAUDIN DE THE).
- Added test for ionex reading ([b6f5585](https://gitlab.com/ydethe/blocksim/commit/b6f55858a10e53fede18d11155a342c770f05cd3) by Yann BLAUDIN DE THE).
- Added doc and cov badges ([1f79ce3](https://gitlab.com/ydethe/blocksim/commit/1f79ce379ad05c7ce940eabc92d78bc396526f66) by Yann BLAUDIN DE THE).

### Fixed

- Fixed pipeline ([caeb37f](https://gitlab.com/ydethe/blocksim/commit/caeb37f3ed73eea2d1d4f5a206f0134ae20667e1) by Yann de Thé).
- Fixed python 3.8 compatibility issue ([524a92f](https://gitlab.com/ydethe/blocksim/commit/524a92fa6878f714b3a51bbd6560db98ee7decd9) by Yann BLAUDIN DE THE).
- Fixed apt install in Gitlab CI ([1b89c68](https://gitlab.com/ydethe/blocksim/commit/1b89c68d88c54c75241fe959d8d617a5bf346379) by Yann BLAUDIN DE THE).
- Fixed pytest conf ([da5a04e](https://gitlab.com/ydethe/blocksim/commit/da5a04e3fca1b22961f220182d3acb9be5b4ae0c) by Yann BLAUDIN DE THE).
- Fixed test compliance ([e9060c7](https://gitlab.com/ydethe/blocksim/commit/e9060c7a9b0ab932f631eaa70dc0ab3762b1d290) by Yann de The).

## [v0.2.0](https://gitlab.com/ydethe/blocksim/tags/v0.2.0) - 2023-02-08

<small>[Compare with v0.1.10](https://gitlab.com/ydethe/blocksim/compare/v0.1.10...v0.2.0)</small>

### Added

- Added Y unit and name in ADSPLine, ave DSPSpectrum in dB by default ([5fbdb5a](https://gitlab.com/ydethe/blocksim/commit/5fbdb5aa9cf4c0859adf04f7e8e23d43eeecd00d) by Yann BLAUDIN DE THE).
- Added all GNSS constellation PRN generation ([c9f670e](https://gitlab.com/ydethe/blocksim/commit/c9f670eee7fe6ed6c2d9f64fb4e3c370eb4ea95b) by Yann BLAUDIN DE THE).
- Added ref image ([bcd8a43](https://gitlab.com/ydethe/blocksim/commit/bcd8a430b2f41f37b672d3411900fc6cbc4334b9) by Yann BLAUDIN DE THE).
- Added DerivativeDSPFilter and DSPRectilinearMap.from_ionex ([dc18b50](https://gitlab.com/ydethe/blocksim/commit/dc18b501ef8d06d5a6a903dc80136bb7c37a7879) by Yann BLAUDIN DE THE).
- Added test for RecursiveSpectrumEstimator ([1390a4f](https://gitlab.com/ydethe/blocksim/commit/1390a4face001cdfd3cae466a8707bc74e36ebcf) by Yann BLAUDIN DE THE).
- Added RecursiveSpectrumEstimator, and renamed SpectrumEstimator to KalmanSpectrumEstimator ([5068302](https://gitlab.com/ydethe/blocksim/commit/50683026bcc56ada81db1ccd9ea1d5a03c37c47e) by Yann BLAUDIN DE THE).
- Added mean_variance_.dgauss_norm ([cead91d](https://gitlab.com/ydethe/blocksim/commit/cead91dc4f2073febd2c5dcff83e74915a67d7d2) by Yann BLAUDIN DE THE).
- Added conda env ([1c21477](https://gitlab.com/ydethe/blocksim/commit/1c21477d7f7e99f64fa5ec6df5b5d41109eb5964) by John Gray).
- Added parquet logger ([3c384a0](https://gitlab.com/ydethe/blocksim/commit/3c384a01c43d4dce60b66e750f1e8800d631ec06) by Yann BLAUDIN DE THE).
- Added sat dist example ([752182c](https://gitlab.com/ydethe/blocksim/commit/752182cafb968f4f5f5a21ff9ad501bc2dd1b8a8) by Yann de The).
- Added combined ToA FoA DOP ([6de4656](https://gitlab.com/ydethe/blocksim/commit/6de46568393940caf89b35ce3cd9027c3cc27433) by Yann BLAUDIN DE THE).
- Added histogram computation and plotting ([e4fa98d](https://gitlab.com/ydethe/blocksim/commit/e4fa98deba3479ad7ccf7f7373dee6f75f665ded) by Yann BLAUDIN DE THE).
- Added images ([dd14074](https://gitlab.com/ydethe/blocksim/commit/dd14074b83756e2cb717efdd5657505f1c0f8025) by Yann BLAUDIN DE THE).
- Added Parameter.getTypeDB ([281f1ce](https://gitlab.com/ydethe/blocksim/commit/281f1ce3f48876cbc629a95aee91a4511238bdd4) by Yann de The).
- Added example, itrf_to_llavpa ([5e5a101](https://gitlab.com/ydethe/blocksim/commit/5e5a101edf30e731197d65a4b67887827d25fa8f) by Yann BLAUDIN DE THE).

### Fixed

- Fixed tests and pdm.lock file ([aa5c7d8](https://gitlab.com/ydethe/blocksim/commit/aa5c7d80b91ea3116d37ec1ecb4a1e6f2fdee777) by Yann de Thé).
- Fixing pipeline ([7549b57](https://gitlab.com/ydethe/blocksim/commit/7549b57e7059c16bd234d2bf93e24fa211763152) by Yann de The).
- Fixed merge ([61ef54a](https://gitlab.com/ydethe/blocksim/commit/61ef54a7048869abe40817e774c47fdb0b98363e) by Yann BLAUDIN DE THE).
- Fixed reqs ([dd96a6f](https://gitlab.com/ydethe/blocksim/commit/dd96a6ff8f7302066a0a6f3cf9228ea0b4cf88a5) by Yann de The).
- Fixed python req ([b1b08ef](https://gitlab.com/ydethe/blocksim/commit/b1b08ef359f7c4d38dd1d511f8b38a2afcdcc5cc) by Yann de The).
- Fixed PlateCarreer BAxe ([0b4a463](https://gitlab.com/ydethe/blocksim/commit/0b4a463993a96f60ddfad58fd2086ca1c7bafea2) by Yann de The).
- Fixed QPSK example ([f3c72b8](https://gitlab.com/ydethe/blocksim/commit/f3c72b8f67cbc756c4568d6d5b18d35b331a4f10) by Yann BLAUDIN DE THE).

### Removed

- Removed LogFormatter.py ([93b41c6](https://gitlab.com/ydethe/blocksim/commit/93b41c68cecdc6eb16526502907c354e4013e0ea) by Yann de The).
- Removed blocksim CLI ([14f3b99](https://gitlab.com/ydethe/blocksim/commit/14f3b99990830d40d8005a939a0c5c90bc809b18) by Yann BLAUDIN DE THE).
- Removed plotBode ([bc0c4fe](https://gitlab.com/ydethe/blocksim/commit/bc0c4fe498151d3edb009a7ea052614903138028) by Yann BLAUDIN DE THE).
- Removed one test ([dc7b1da](https://gitlab.com/ydethe/blocksim/commit/dc7b1da96c2572d84b5eaba4dd10adc9cf07caf1) by Yann BLAUDIN DE THE).

## [v0.1.10](https://gitlab.com/ydethe/blocksim/tags/v0.1.10) - 2022-05-13

<small>[Compare with v0.1.8](https://gitlab.com/ydethe/blocksim/compare/v0.1.8...v0.1.10)</small>

### Added

- Added tests ([944711f](https://gitlab.com/ydethe/blocksim/commit/944711f78389e39e20b298fb19acd9f1f1ad33bb) by Yann BLAUDIN DE THE).

## [v0.1.8](https://gitlab.com/ydethe/blocksim/tags/v0.1.8) - 2022-04-21

<small>[Compare with v0.1.7](https://gitlab.com/ydethe/blocksim/compare/v0.1.7...v0.1.8)</small>

## [v0.1.7](https://gitlab.com/ydethe/blocksim/tags/v0.1.7) - 2022-04-21

<small>[Compare with v0.1.5](https://gitlab.com/ydethe/blocksim/compare/v0.1.5...v0.1.7)</small>

### Added

- Added group_delays ([86518dc](https://gitlab.com/ydethe/blocksim/commit/86518dc453cedfe397ea1ae079c7ef5c7868f0de) by Yann de The).
- Added tests ([1783a60](https://gitlab.com/ydethe/blocksim/commit/1783a602053ce68e2b680d6fe71a9407867c0290) by Yann de The).
- Added filter tests ([1851e10](https://gitlab.com/ydethe/blocksim/commit/1851e1096ad9dde327eda32d89b78c5f0e8df7c2) by Yann de Thé).
- Added gitalb CI ([cc84be5](https://gitlab.com/ydethe/blocksim/commit/cc84be58583baf803e8ab093c6d337904515e786) by Yann de The).

### Removed

- Removed group_delays.py ([11b4895](https://gitlab.com/ydethe/blocksim/commit/11b48959ee87cd4bd1fa84ab259557bc4741db43) by Yann de The).

## [v0.1.5](https://gitlab.com/ydethe/blocksim/tags/v0.1.5) - 2022-04-14

<small>[Compare with v0.1.4](https://gitlab.com/ydethe/blocksim/compare/v0.1.4...v0.1.5)</small>

### Added

- Added gitalb CI ([19c978e](https://gitlab.com/ydethe/blocksim/commit/19c978e06b9d80aca875d2ecd960c4897f012fbe) by Yann de The).
- Added conda env ([9f19048](https://gitlab.com/ydethe/blocksim/commit/9f19048107077e7414983e36df0b2763f8865930) by Yann de The).
- Added propag tuto ([2bbf928](https://gitlab.com/ydethe/blocksim/commit/2bbf928ef2c67b008ff4490d18af584b93dfccb9) by Yann de Thé).

## [v0.1.4](https://gitlab.com/ydethe/blocksim/tags/v0.1.4) - 2022-04-12

<small>[Compare with v0.1.3](https://gitlab.com/ydethe/blocksim/compare/v0.1.3...v0.1.4)</small>

### Added

- Added quadcopter example ([0fac20b](https://gitlab.com/ydethe/blocksim/commit/0fac20b964da428df587baca5a2ec197f1633a55) by YANN BLAUDIN DE THE).

## [v0.1.3](https://gitlab.com/ydethe/blocksim/tags/v0.1.3) - 2022-04-12

<small>[Compare with first commit](https://gitlab.com/ydethe/blocksim/compare/046bce3dffeba654ee926ca331056dd6848d2132...v0.1.3)</small>

### Added

- Added example generation ([192fd16](https://gitlab.com/ydethe/blocksim/commit/192fd1638ccc81d6c66b974c3382836ab6a0afa1) by YANN BLAUDIN DE THE).
- Added notebook ([c883bb4](https://gitlab.com/ydethe/blocksim/commit/c883bb42b0007bbbd38a438e527ed262fd338297) by Yann BLAUDIN DE THE).
- Added baseline ([8a2bfd1](https://gitlab.com/ydethe/blocksim/commit/8a2bfd197887bd2ce44e3025cc98d49c63c6d978) by Yann Blaudin De The).
- Added doc ([ac838b0](https://gitlab.com/ydethe/blocksim/commit/ac838b05824baa185942c9e6546f7cf2e64dc41e) by Yann Blaudin De The).
- Added dsp func ([81c9427](https://gitlab.com/ydethe/blocksim/commit/81c942706df24eb8360bd84bf8555e724dcc4dcd) by YANN BLAUDIN DE THE).
- Added reqs ([85b73ea](https://gitlab.com/ydethe/blocksim/commit/85b73eae08d796ce40190c9be8b076d47f6bca64) by YANN BLAUDIN DE THE).
- Added pluggy ([3354edf](https://gitlab.com/ydethe/blocksim/commit/3354edf6925d6257a95026338436432daf0b4c70) by Yann BLAUDIN DE THE).
- Added test_result.xml to .gitignore ([f0f699d](https://gitlab.com/ydethe/blocksim/commit/f0f699d4328c4310158d3933e92b22c5f93b5303) by Yann BLAUDIN DE THE).
- Added Cobertra coverage results ([05f081c](https://gitlab.com/ydethe/blocksim/commit/05f081ce8943d1d304629dfcf2808cc79d66d712) by Yann BLAUDIN DE THE).
- Added JUnit XML test results ([83ac421](https://gitlab.com/ydethe/blocksim/commit/83ac421495238805681debe9359d259c54dd575b) by Yann BLAUDIN DE THE).
- Added RTPlotter ([9ed7b78](https://gitlab.com/ydethe/blocksim/commit/9ed7b7873b43c91b68b8a00d51a01a58fc5f43d2) by Yann BLAUDIN DE THE).
- Added CircleCI conf ([dc7b875](https://gitlab.com/ydethe/blocksim/commit/dc7b8755771dc42862e480ba29771f5fd4055bdf) by Yann BLAUDIN DE THE).
- Added plot_radar.py ([feed014](https://gitlab.com/ydethe/blocksim/commit/feed014b479cfffcee3def80cf3f5f97aa45b075) by Yann BLAUDIN DE THE).
- Added baseline images ([187924c](https://gitlab.com/ydethe/blocksim/commit/187924c229a3060a1b18e8ae8f3c7cbe953889ae) by Yann BLAUDIN DE THE).
- Added high order derivation for DSPLine ([10377b0](https://gitlab.com/ydethe/blocksim/commit/10377b070734f10670f789fa917359d69f489158) by Yann BLAUDIN DE THE).
- Added PostGreSQL logging ([4ddec17](https://gitlab.com/ydethe/blocksim/commit/4ddec1745a5a6ed6e01ec1351be0c7c67b594b9e) by Yann BLAUDIN DE THE).
- Added logged attribute on Computer, that allows disabling logging. Correcting binary logging ([8953385](https://gitlab.com/ydethe/blocksim/commit/895338535e185763c85f6910b7acbf0619c0fbe4) by Yann BLAUDIN DE THE).
- Added ubuntu docker image (commented) for memry ([0a87f85](https://gitlab.com/ydethe/blocksim/commit/0a87f8571831fec311ad1037c3491cd126d1885c) by Yann BLAUDIN DE THE).
- Added lazy props for DSPLine interpolators ([5ddcc8c](https://gitlab.com/ydethe/blocksim/commit/5ddcc8c8daa1ff50f88a113e5f3bb6c6d886df39) by Yann BLAUDIN DE THE).
- Added cartopy ([d914bf2](https://gitlab.com/ydethe/blocksim/commit/d914bf2d62a0fd672dade348a8a53fa0c25be0e8) by Yann BLAUDIN DE THE).
- Added links in pypi metadata ([cdc5d5a](https://gitlab.com/ydethe/blocksim/commit/cdc5d5a772cd93e61d387c21f24502c3ef25c187) by Yann BLAUDIN DE THE).
- Added header comments ([568eb8e](https://gitlab.com/ydethe/blocksim/commit/568eb8e0dfa708a6b79506da6f2e35b9f430b3c4) by Yann BLAUDIN DE THE).
- Added auto unit prefix ([be07c70](https://gitlab.com/ydethe/blocksim/commit/be07c70a1485f3c695670bcbd55f6cf493c6cc07) by Yann BLAUDIN DE THE).
- Added extensions in README ([ab839e9](https://gitlab.com/ydethe/blocksim/commit/ab839e9293fe217bb81bf49bb9b0322fe2b316b2) by Yann BLAUDIN DE THE).
- Added complex logging ([f5ccd6e](https://gitlab.com/ydethe/blocksim/commit/f5ccd6e7876af9f4fe5dc7b3da7750641cac9e23) by Yann BLAUDIN DE THE).
- Added BOC test images ([8105661](https://gitlab.com/ydethe/blocksim/commit/8105661d1b8184da92876e5e4a53a4ba7a728f06) by Yann BLAUDIN DE THE).
- Added BOC modulator ([c29a8d0](https://gitlab.com/ydethe/blocksim/commit/c29a8d07856c61dfbf8227f45af364da1787b72a) by Yann BLAUDIN DE THE).
- Added examples in doc, test QPSK, DSPFilter as AComputer ([51a46bb](https://gitlab.com/ydethe/blocksim/commit/51a46bb275954e201f0c4f4587cb7aae73fd8bff) by Yann BLAUDIN DE THE).
- Added ref images Gold sequences ([b9dedd1](https://gitlab.com/ydethe/blocksim/commit/b9dedd1f1b2114310cca42d80b8c1ef512288a5f) by Yann BLAUDIN DE THE).
- Added integration ([1bae2ee](https://gitlab.com/ydethe/blocksim/commit/1bae2ee6d60ec3cb7cc96099f29661084cc47644) by Yann BLAUDIN DE THE).
- Added ground track example ([98d000c](https://gitlab.com/ydethe/blocksim/commit/98d000c0e8029b730fca50e6c85b5be1e2dc82da) by Yann BLAUDIN DE THE).
- Added examples. Removed shape_meas argument in estimators ([3fb33b7](https://gitlab.com/ydethe/blocksim/commit/3fb33b74e2fee0d228a77f989ae61c6550f7fd7b) by Yann BLAUDIN DE THE).
- Addind test StreamCSVSensors ([5e22eeb](https://gitlab.com/ydethe/blocksim/commit/5e22eeb739ed284c4553fb147508532c99e8bb8a) by Yann BLAUDIN DE THE).

### Fixed

- Fixed Kalman example ([71f0266](https://gitlab.com/ydethe/blocksim/commit/71f0266f55e47fea20b2419ef750147401645937) by Yann BLAUDIN DE THE).
- Fixed version string ([5ed787b](https://gitlab.com/ydethe/blocksim/commit/5ed787ba3eb3e7c59b541bcdc4363a5238794b82) by Yann BLAUDIN DE THE).
- Fix sqlalchemy import ([2cfd20f](https://gitlab.com/ydethe/blocksim/commit/2cfd20f21a96d27606ddcc3124984268bb978d96) by Yann BLAUDIN DE THE).
- Fix blocksim db call in Gitlab CI ([8c6b772](https://gitlab.com/ydethe/blocksim/commit/8c6b77285b8bd918b50764cd8bd2c94c31725ef2) by Yann BLAUDIN DE THE).
- Fix tests ([484bfcf](https://gitlab.com/ydethe/blocksim/commit/484bfcf5bd2826a50b92c1dccf9bacb2f9917cfb) by Yann BLAUDIN DE THE).
- Fix reqs ([d70f227](https://gitlab.com/ydethe/blocksim/commit/d70f227ff1d6da489bacd4cc6cee91730bf155d1) by Yann BLAUDIN DE THE).
- Fix black version in .pre-commit-config.yaml ([cc1fed4](https://gitlab.com/ydethe/blocksim/commit/cc1fed4898e81df1c5e3b0d5b66b56fe7fe6ab34) by Yann BLAUDIN DE THE).
- Fix test ([b69e68f](https://gitlab.com/ydethe/blocksim/commit/b69e68fc63bca477ef01afe96047fcde6f3059ac) by Yann BLAUDIN DE THE).
- Fixed conflict ([8d5db89](https://gitlab.com/ydethe/blocksim/commit/8d5db891c3e69b78d7ecaefca0da5a7acc5e115e) by Yann BLAUDIN DE THE).
- Fix conflict ([54028cd](https://gitlab.com/ydethe/blocksim/commit/54028cd24055019fd8e556a54a869210f836ae86) by Yann BLAUDIN DE THE).
- Fix CI pipeline ([21a8a68](https://gitlab.com/ydethe/blocksim/commit/21a8a68daed1825274bdde0ec9d2670d717920fe) by Yann BLAUDIN DE THE).
- Fix examples ([74ddad8](https://gitlab.com/ydethe/blocksim/commit/74ddad87b35acd4b7c4e005f35d7a52187754d51) by Yann BLAUDIN DE THE).
- Fix docs/environment.yml ([6402728](https://gitlab.com/ydethe/blocksim/commit/640272867d65073ed195908ce4678471e3c93855) by Yann BLAUDIN DE THE).
- Fix tests/baseline/test_dsp_setpoint.png ([1ea07b8](https://gitlab.com/ydethe/blocksim/commit/1ea07b8085487b4b3950c2d1bf12393e7d6dbc47) by Yann BLAUDIN DE THE).
- Fix environment.yml ([e73e1a4](https://gitlab.com/ydethe/blocksim/commit/e73e1a4bc51fd403ccabe8e98c2cb6ba7f83f6c3) by Yann BLAUDIN DE THE).

### Removed

- Removed doc ([c6fffff](https://gitlab.com/ydethe/blocksim/commit/c6fffff74785908cbf3260bd2e0349958bb83503) by Yann Blaudin De The).
- Removed old references to blocksim_sigspace ([6743c37](https://gitlab.com/ydethe/blocksim/commit/6743c370d5519b739da9232ce9ed6239de4a750d) by YANN BLAUDIN DE THE).
- Removed codecov ([e447bfc](https://gitlab.com/ydethe/blocksim/commit/e447bfc344a9748434e8767a739bfa8ced33c863) by Yann BLAUDIN DE THE).
- Removed OrderedDict ([f0de4de](https://gitlab.com/ydethe/blocksim/commit/f0de4de82ea1ef89ff6155784704d5793729ab6d) by Yann BLAUDIN DE THE).

